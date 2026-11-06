import os
import argparse
import hashlib
import random
import numpy as np
import torch
import torch.distributed as dist
from exp.exp_basic import get_available_model_names, get_default_model_name
from exp.exp_forecast import Exp_Forecast
from utils.tools import ForecastCheckpointError


def compact_path_component(value, max_len=120):
    value = str(value)
    if len(value) <= max_len:
        return value
    digest = hashlib.sha1(value.encode('utf-8')).hexdigest()[:8]
    keep_len = max(max_len - len(digest) - 2, 1)
    prefix = value[:keep_len].rstrip('_-.')
    return f'{prefix}_h{digest}'


def build_setting(args, run_idx):
    payload = '|'.join(
        f'{key}={repr(value)}'
        for key, value in sorted(vars(args).items())
        if key not in {'local_rank', 'device_ids'}
    )
    setting_hash = hashlib.sha1(f'{payload}|run_idx={run_idx}'.encode('utf-8')).hexdigest()[:8]
    des = compact_path_component(args.des)
    return (
        f'{args.task_name}_{args.model_id}_{args.model}'
        f'_sl{args.seq_len}_it{args.input_token_len}_ot{args.output_token_len}'
        f'_{des}_h{setting_hash}_{run_idx}'
    )


if __name__ == '__main__':
    available_models = get_available_model_names()
    parser = argparse.ArgumentParser(description='Forecast experiment runner')

    # basic config
    parser.add_argument('--task_name', type=str, default='forecast', choices=['forecast'], help='task name')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--skip_test', action='store_true', help='train and select a checkpoint on validation only; do not evaluate test')
    parser.add_argument('--monitor_test_during_training', action='store_true', help='explicitly opt into test curves during training; not used for selection')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, default=get_default_model_name(), choices=available_models,
                        help='model name')
    parser.add_argument('--seed', type=int, default=2021, help='seed')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', choices=['M', 'S', 'MS'], help='forecasting task type')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--results_root', type=str, default='./test_results/', help='isolated evaluation output root')
    parser.add_argument('--test_flag', type=str, default='T', help='test domain')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=672, help='input sequence length')
    parser.add_argument('--input_token_len', type=int, default=576, help='input token length')
    parser.add_argument('--output_token_len', type=int, default=96, help='output token length')
    parser.add_argument('--test_seq_len', type=int, default=672, help='test seq len')
    parser.add_argument('--test_pred_len', type=int, default=96, help='test pred len')

    # model define
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--e_layers', type=int, default=1, help='encoder layers')
    parser.add_argument('--d_model', type=int, default=512, help='d model')
    parser.add_argument('--n_heads', type=int, default=8, help='n heads')
    parser.add_argument('--d_ff', type=int, default=2048, help='d ff')
    parser.add_argument('--activation', type=str, default='relu', help='activation')
    parser.add_argument('--covariate', action='store_true', help='use cov', default=False)
    parser.add_argument('--node_num', type=int, default=100, help='number of nodes')
    parser.add_argument('--use_norm', action='store_true', help='use norm', default=False)
    parser.add_argument('--nonautoregressive', action='store_true', help='nonautoregressive', default=False)
    parser.add_argument('--test_dir', type=str, default='./test', help='test dir')
    parser.add_argument('--test_file_name', type=str, default='checkpoint.pth', help='test file')
    parser.add_argument('--output_attention', action='store_true', help='output attention', default=False)
    parser.add_argument('--visualize', action='store_true', help='visualize', default=False)
    parser.add_argument('--flash_attention', action='store_true', help='flash attention', default=False)
    parser.add_argument('--use_point_representation', action='store_true',
                        help='use observation-point representation frontend', default=False)
    parser.add_argument('--point_hidden_dim', type=int, default=128,
                        help='hidden dimension of observation-point representation frontend')
    parser.add_argument('--point_layers', type=int, default=2,
                        help='deprecated compatibility option; CausalODE point frontend no longer uses neighborhood propagation layers')
    parser.add_argument('--point_knn', type=int, default=8,
                        help='deprecated compatibility option; no longer controls point frontend propagation, only used as fallback for point_local_fusion_k')
    parser.add_argument('--point_neighbor_window', type=int, default=-1,
                        help='deprecated compatibility option; point frontend density features now use input_token_len')
    parser.add_argument('--point_token_subbins', type=int, default=4,
                        help='number of temporal sub-bins used inside each token during point-to-token aggregation')
    parser.add_argument('--point_time_dim', type=int, default=32, help='hidden dimension of point time encoder')
    parser.add_argument('--point_local_fusion_mode', type=str, default='dual_gate',
                        choices=['none', 'last', 'time_weighted', 'dual_gate', 'causal_time_weighted'],
                        help='query-side local prediction mode used by the gated local fusion head')
    parser.add_argument('--point_force_fusion', type=str, default='auto',
                        choices=['auto', 'global', 'local', 'strict_local', 'feature'],
                        help='ablation switch for AdaLG query fusion: auto uses gated output fusion, feature uses gated feature-level fusion, global ignores local predictions, local uses local predictions with global fallback, strict_local removes the global fallback')
    parser.add_argument('--point_disable_local_branch', action='store_true',
                        help='remove query-side local prediction and fusion parameters for parameter-matched global-only controls',
                        default=False)
    parser.add_argument('--point_fixed_gate', type=float, default=None,
                        help='use a fixed local fusion gate in [0, 1] for fixed-weight fusion ablations; unset keeps the learned adaptive gate')
    parser.add_argument('--point_prediction_variant', choices=['adalg', 'raw_recent_concat'], default='adalg')
    parser.add_argument('--point_recent_strict_past', action='store_true', default=False)
    parser.add_argument('--raw_concat_hidden', type=int, default=0, help='0 matches the AdaLG query-head parameter budget')
    parser.add_argument('--raw_recent_query_export', action='store_true', default=False)
    parser.add_argument('--point_local_branch_source', choices=['raw', 'shared_encoder'], default='raw',
                        help='raw: independent recent target observations; shared_encoder: second head on the global encoder context')
    parser.add_argument('--point_local_encoder', choices=['summary', 'attention'], default='summary',
                        help='independent raw local encoder: existing summary refiner or learned recent-observation attention')
    parser.add_argument('--point_local_fusion_k', type=int, default=8,
                        help='number of recent same-variable history points used by the query-side local prediction branch')
    parser.add_argument('--point_local_gate_bias', type=float, default=-4.0,
                        help='initial bias of the gated local fusion head; more negative keeps the model closer to the original global head at initialization')
    parser.add_argument('--point_dual_gate_bias', type=float, default=-1.0,
                        help='initial bias of the dual-gate fusion head used when point_local_fusion_mode=dual_gate')
    parser.add_argument('--point_gate_max', type=float, default=1.0,
                        help='maximum local-branch fusion gate; applied as point_gate_max * sigmoid(logit)')
    parser.add_argument('--point_gate_mode', type=str, default='utility',
                        choices=['ordinary', 'utility', 'fixed', 'global', 'local'],
                        help='query-level fusion baseline; utility adds oracle-gate and local-advantage supervision')
    parser.add_argument('--point_train_stage', type=str, default='full',
                        choices=['full', 'branch', 'gate'],
                        help='full uses the original train set; branch/gate use disjoint subsets of that train set')
    parser.add_argument('--point_gate_holdout_ratio', type=float, default=0.0,
                        help='fraction of original training groups reserved for gate-only training')
    parser.add_argument('--point_gate_split_seed', type=int, default=1729,
                        help='seed offset for the reproducible branch/gate holdout')
    parser.add_argument('--point_gate_init_checkpoint', type=str, default='',
                        help='AdaLG gate-only fine-tuning: load a pretrained two-branch checkpoint, freeze all other parameters, and train only the gate; supports {seed} and a unique glob pattern')
    parser.add_argument('--point_gate_initial_weight', type=float, default=None,
                        help='actual initial gate weight for frozen gate training; must be smaller than point_gate_max')
    parser.add_argument('--point_frozen_global_local_training', action='store_true', default=False,
                        help='gate holdout stage: load a Global-only source and train Local plus gate, keeping every Global tensor frozen')
    parser.add_argument('--point_gate_prediction_features', action='store_true', default=False,
                        help='append global/local predictions, their difference, and absolute difference to the AdaLG gate input')
    parser.add_argument('--point_separate_utility_head', action='store_true', default=False,
                        help='use distinct utility-classification and fusion logits in the same parameter-matched gate; both contribute to fusion')
    parser.add_argument('--point_disable_causal_gate_context', action='store_true', default=False,
                        help='zero the causal context supplied to the local/global gate while keeping parameter count fixed')
    parser.add_argument('--point_disable_gate_availability_features', action='store_true', default=False,
                        help='zero the gate availability-statistic inputs for an information-source-only ordinary-gate control')
    parser.add_argument('--global_aux_loss_weight', type=float, default=0.25,
                        help='auxiliary global-branch prediction loss weight')
    parser.add_argument('--local_aux_loss_weight', type=float, default=0.25,
                        help='auxiliary local-branch prediction loss weight')
    parser.add_argument('--utility_loss_weight', type=float, default=0.1,
                        help='binary local-better utility supervision weight; active only in utility gate mode')
    parser.add_argument('--oracle_gate_loss_weight', type=float, default=0.1,
                        help='oracle bounded-gate regression weight; active only in utility gate mode')
    parser.add_argument('--point_causal_context_mode', type=str, default='normal',
                        choices=['normal', 'zero', 'shuffled'],
                        help='causal context ablation mode: normal uses CausalODE context, zero passes zeros into dual_gate, shuffled breaks batch/variable alignment')
    parser.add_argument('--point_use_topology', action='store_true',
                        help='deprecated compatibility option; topology bias is now generated by CausalODE',
                        default=False)
    parser.add_argument('--point_use_node_embedding', action='store_true',
                        help='use explicit node-id embeddings in point frontend/head when batches provide obs_node_id/query_node_id',
                        default=False)
    parser.add_argument('--causal_ode_state_dim', type=int, default=32,
                        help='latent state dimension of constrained CausalODE graph generator')
    parser.add_argument('--causal_ode_hidden', type=int, default=128,
                        help='hidden dimension of constrained CausalODE graph generator')
    parser.add_argument('--causal_ode_constraint_strength', type=float, default=0.0,
                        help='strength of CausalODE manifold correction')
    parser.add_argument('--causal_ode_h_weight', type=float, default=0.001,
                        help='weight of CausalODE DAG acyclicity penalty')
    parser.add_argument('--causal_ode_l1_weight', type=float, default=0.0001,
                        help='weight of CausalODE adjacency L1 penalty')
    parser.add_argument('--causal_ode_diag_weight', type=float, default=0.001,
                        help='weight of CausalODE diagonal-zero penalty')
    parser.add_argument('--causal_ode_laplacian_weight', type=float, default=0.0001,
                        help='weight of CausalODE temporal smoothness penalty')
    parser.add_argument('--causal_ode_adjacency_scale', type=float, default=1.0,
                        help='scale applied to CausalODE adjacency messages')
    parser.add_argument('--causal_ode_gate_init', type=float, default=-4.0,
                        help='initial residual gate logit for CausalODE message injection')
    parser.add_argument('--causal_ode_steps', type=int, default=8,
                        help='number of ODE integration steps (replaces full seq_len integration; 8 is fast and sufficient)')
    parser.add_argument('--causal_prior_window', type=int, default=30,
                        help='recent history window used to average CausalODE adjacency into [B,V,V] causal_prior')
    parser.add_argument('--causal_prior_mode', type=str, default='normal',
                        choices=['normal', 'zero', 'shuffled'],
                        help='causal prior ablation mode used by causal_time_weighted local fusion')
    parser.add_argument('--causal_local_alpha', type=float, default=0.0,
                        help='causal prior logit weight used by causal_time_weighted local fusion')
    parser.add_argument('--causal_local_beta', type=float, default=1.0,
                        help='same-variable bonus used by causal_time_weighted local fusion when alpha > 0')
    parser.add_argument('--causal_local_expand_pool', action='store_true', default=False,
                        help='expand causal_time_weighted local candidates from same variable to all variables in the local time window')
    parser.add_argument('--point_val_query_mode', type=str, default='default',
                        choices=['default', 'dense_autoregressive', 'future_only'],
                        help='query construction mode for validation point batches')
    parser.add_argument('--point_test_query_mode', type=str, default='future_only',
                        choices=['default', 'dense_autoregressive', 'future_only'],
                        help='query construction mode for test point batches')
    parser.add_argument('--point_align_val_with_test', action='store_true',
                        help='use test horizon and query mode for validation point batches', default=False)
    parser.add_argument('--point_train_size', type=float, default=None,
                        help='point dataset train split ratio; default uses dataset class setting')
    parser.add_argument('--point_val_size', type=float, default=None,
                        help='point dataset validation split ratio; default uses dataset class setting')
    parser.add_argument('--point_test_size', type=float, default=None,
                        help='point dataset test split ratio; default uses dataset class setting')
    parser.add_argument('--point_split_mode', type=str, default='random',
                        choices=['random', 'record', 'chronological', 'reverse_chronological'],
                        help='point dataset split protocol')
    parser.add_argument('--point_stratified_random_split', action='store_true',
                        help='stratify random point splits by target difficulty', default=False)
    parser.add_argument('--point_stratify_bins', type=int, default=10,
                        help='number of target difficulty buckets for stratified random point splits')
    parser.add_argument('--inspect_point_splits', action='store_true',
                        help='print point split target/mask/time statistics for train/val/test', default=False)
    parser.add_argument('--astgi_channel_dim', '--stppgn_channel_dim', dest='astgi_channel_dim', type=int, default=24,
                        help='ASTGI channel embedding dimension')
    parser.add_argument('--astgi_time_dim', '--stppgn_time_dim', dest='astgi_time_dim', type=int, default=24,
                        help='ASTGI time embedding dimension')
    parser.add_argument('--astgi_prop_layers', '--stppgn_prop_layers', dest='astgi_prop_layers', type=int, default=3,
                        help='deprecated compatibility option; ASTGI uses CausalODE propagation')
    parser.add_argument('--astgi_k_neighbors', '--stppgn_k_neighbors', dest='astgi_k_neighbors', type=int, default=80,
                        help='deprecated compatibility option; ASTGI no longer uses KNN')
    parser.add_argument('--astgi_mlp_ratio', '--stppgn_mlp_ratio', dest='astgi_mlp_ratio', type=float, default=2.0,
                        help='ASTGI MLP expansion ratio')
    parser.add_argument('--astgi_channel_dist_weight', '--stppgn_channel_dist_weight', dest='astgi_channel_dist_weight',
                        type=float, default=1.0,
                        help='deprecated compatibility option; ASTGI attention no longer applies a channel-distance weight')
    parser.add_argument('--astgi_query_chunk_size', type=int, default=1024,
                        help='ASTGI query chunk size for full-history attention')
    parser.add_argument('--tpatch_patch_size', type=float, default=0.0,
                        help='tPatchGNN patch time window; <=0 falls back to input_token_len')
    parser.add_argument('--tpatch_stride', type=float, default=0.0,
                        help='tPatchGNN patch stride; <=0 falls back to tpatch_patch_size')
    parser.add_argument('--tpatch_hop', type=int, default=1, help='tPatchGNN graph convolution hop count')
    parser.add_argument('--tpatch_tf_layer', type=int, default=1,
                        help='number of Transformer layers inside each tPatchGNN temporal encoder')
    parser.add_argument('--tpatch_n_layer', type=int, default=1,
                        help='number of stacked tPatchGNN temporal/GNN layers')
    parser.add_argument('--tpatch_te_dim', type=int, default=10, help='tPatchGNN time encoding dimension')
    parser.add_argument('--tpatch_node_dim', type=int, default=10, help='tPatchGNN adaptive node embedding dimension')
    parser.add_argument('--tpatch_hid_dim', type=int, default=64, help='tPatchGNN hidden dimension')
    parser.add_argument('--tpatch_n_heads', type=int, default=1, help='tPatchGNN Transformer attention heads')
    parser.add_argument('--tpatch_outlayer', type=str, default='Linear', choices=['Linear', 'CNN'],
                        help='tPatchGNN temporal output aggregation layer')
    parser.add_argument('--dlinear_moving_avg', type=int, default=25, help='DLinear moving average kernel size')
    parser.add_argument('--dlinear_individual', action='store_true',
                        help='use one DLinear seasonal/trend projection per variable', default=False)
    parser.add_argument('--grud_hidden_size', type=int, default=100, help='GRU-D hidden state size')
    parser.add_argument('--grafiti_latent_dim', type=int, default=128, help='GraFITi latent dimension')
    parser.add_argument('--grafiti_n_layers', type=int, default=2, help='number of GraFITi message passing layers')
    parser.add_argument('--grafiti_attn_heads', type=int, default=4, help='number of GraFITi attention heads')
    parser.add_argument('--timecheat_latent_dim', type=int, default=128, help='TimeCHEAT graph latent dimension')
    parser.add_argument('--timecheat_n_layers', type=int, default=2, help='number of TimeCHEAT graph layers')
    parser.add_argument('--timecheat_attn_heads', type=int, default=4, help='number of TimeCHEAT graph attention heads')
    parser.add_argument('--timecheat_ref_points', type=int, default=128, help='number of TimeCHEAT reference points')
    parser.add_argument('--timecheat_n_patches', type=int, default=8, help='number of TimeCHEAT temporal patches')
    parser.add_argument('--timecheat_former_layers', type=int, default=3, help='number of TimeCHEAT Transformer layers')
    parser.add_argument('--timecheat_former_heads', type=int, default=8, help='number of TimeCHEAT Transformer heads')
    parser.add_argument('--timecheat_former_dff', type=int, default=256, help='TimeCHEAT Transformer feed-forward size')
    parser.add_argument('--mtan_num_ref_points', type=int, default=64, help='number of mTAN reference points')
    parser.add_argument('--mtan_latent_dim', type=int, default=16, help='mTAN latent dimension')
    parser.add_argument('--mtan_rec_hidden', type=int, default=64, help='mTAN encoder RNN hidden size')
    parser.add_argument('--mtan_gen_hidden', type=int, default=50, help='mTAN decoder RNN hidden size')
    parser.add_argument('--mtan_embed_time', type=int, default=128, help='mTAN time embedding dimension')
    parser.add_argument('--mtan_num_heads', type=int, default=4, help='number of mTAN time-attention heads')
    parser.add_argument('--mtan_k_samples', type=int, default=1, help='number of mTAN latent samples averaged per forward')
    parser.add_argument('--mtan_kl_weight', type=float, default=0.0, help='optional mTAN latent KL regularization weight')
    # adaptation
    parser.add_argument('--adaptation', action='store_true', help='adaptation', default=False)
    parser.add_argument('--pretrain_model_path', type=str, default='pretrain_model.pth', help='pretrain model path')
    parser.add_argument('--adaptation_load_mode', type=str, default='full', choices=['full', 'compatible', 'backbone'],
                        help='checkpoint loading mode for adaptation')
    parser.add_argument('--subset_rand_ratio', type=float, default=1, help='few shot ratio')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'], help='optimizer type')
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'mae', 'huber', 'mae_mse'],
                        help='masked prediction loss used for training')
    parser.add_argument('--checkpoint_metric', type=str, default='mse',
                        choices=['mse', 'mae', 'mae_mse', 'val_mse', 'val_mae', 'val_mae_mse'],
                        help='validation metric used for early stopping and checkpoint selection')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--emit_loss_history', action='store_true',
                        help='emit machine-readable loss history to stdout for shell aggregation', default=False)
    parser.add_argument('--query_evidence_analysis', action='store_true',
                        help='collect query-wise local evidence and gate statistics during test', default=False)
    parser.add_argument('--gate_split_diagnostics_only', action='store_true',
                        help='load an existing gate checkpoint and report gate-train versus validation calibration without testing or retraining')
    parser.add_argument('--query_evidence_out_dir', type=str, default='',
                        help='output directory for query evidence analysis; defaults to the test result folder')
    parser.add_argument('--query_evidence_density_key', type=str, default='local_density',
                        choices=['local_density', 'candidate_count', 'valid_history_ratio', 'nearest_gap'],
                        help='query evidence variable used to form sparse/medium/dense bins')
    parser.add_argument('--query_evidence_bins', type=int, default=3,
                        help='number of density bins for query evidence analysis')
    parser.add_argument('--query_evidence_save_raw', action='store_true',
                        help='save per-query evidence rows in addition to aggregate summaries', default=False)
    parser.add_argument('--query_evidence_max_raw_rows', type=int, default=200000,
                        help='maximum raw query evidence rows to save; <=0 saves all rows')
    parser.add_argument('--local_advantage_comparable_quantile', type=float, default=0.2,
                        help='fraction of has-local queries with the smallest |local advantage| assigned to the Comparable group')
    parser.add_argument('--eval_local_observation_drop_rate', type=float, default=0.0,
                        help='test-time random deletion rate for local history observations used by the query local branch')
    parser.add_argument('--eval_local_observation_drop_seed', type=int, default=0,
                        help='base seed for test-time local observation deletion; 0 falls back to --seed')
    parser.add_argument('--local_observation_drop_robustness', action='store_true',
                        help='after the standard test pass, evaluate robustness under several local observation deletion rates',
                        default=False)
    parser.add_argument('--input_observation_drop_robustness', action='store_true',
                        help='remove observations before encoder, causal propagation, local prediction, and gate evaluation',
                        default=False)
    parser.add_argument('--local_observation_drop_rates', type=str, default='0,0.25,0.5,0.75',
                        help='comma-separated test-time local observation deletion rates for robustness analysis')
    parser.add_argument('--local_degradation_modes', default='drop',
                        help='comma-separated local-only interventions: drop, recent_drop, noise, stale; severity uses local_observation_drop_rates')
    parser.add_argument('--overfit_eval_train_loss', action='store_true',
                        help='compute the overfit train-loss curve in eval mode so it is comparable with val/test loss',
                        default=False)
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--cosine', action='store_true', help='use cosine annealing lr', default=False)
    parser.add_argument('--tmax', type=int, default=10, help='tmax in cosine anealing lr')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--valid_last', action='store_true', help='valid last', default=False)
    parser.add_argument('--last_token', action='store_true', help='last token', default=False)

    # GPU
    parser.add_argument('--gpu', type=int, default=2, help='gpu')
    parser.add_argument('--ddp', action='store_true', help='Distributed Data Parallel', default=False)
    parser.add_argument('--dp', action='store_true', help='Data Parallel', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()
    if args.gate_split_diagnostics_only and args.is_training:
        parser.error('--gate_split_diagnostics_only requires --is_training 0.')
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    if args.dp:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    elif args.ddp:
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "64209")
        hosts = int(os.environ.get("WORLD_SIZE", "8"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        gpus = torch.cuda.device_count()
        args.local_rank = local_rank
        print(ip, port, hosts, rank, local_rank, gpus)
        dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts,
                                rank=rank)
        torch.cuda.set_device(local_rank)

    if args.task_name == 'forecast':
        Exp = Exp_Forecast
    else:
        Exp = Exp_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = build_setting(args, ii)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            if not args.skip_test and not args.ddp and not args.dp:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = build_setting(args, ii)
        exp = Exp(args)  # set experiments
        try:
            exp.test(setting, test=1)
        except ForecastCheckpointError as exc:
            print(f'[checkpoint_incompatible] {exc}', flush=True)
            # The benchmark runner archives the incompatible run and retrains
            # only this seed. Other failures retain their normal error status.
            raise SystemExit(86) from None
        torch.cuda.empty_cache()
