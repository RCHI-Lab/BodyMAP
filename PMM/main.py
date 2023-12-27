from PMMTrainer import PMMTrainer
from WSTrainer import WSTrainer
import os
from constants import MODALITY_TO_FEATURE_SIZE, BASE_PATH
import json
import argparse


def load_args_from_json(json_file):
    try:
        with open(json_file, 'r') as file:
            arguments = json.load(file)
            return arguments
    except FileNotFoundError:
        print(f"File '{json_file}' not found.")
        return None
    except json.JSONDecodeError as exc:
        print(f"Error decoding JSON: {exc}")
        return None


def parse_args(args):
    args['data_path'] = os.path.join(BASE_PATH, "BodyPressure/data_BP")
    args['save_path'] = os.path.join(BASE_PATH, "PMM_exps")

    # data setup
    if args['exp_type'] == 'overfit':
        args['synth_train_file'] = os.path.join(BASE_PATH, "BodyMAP/data_files/synth_overfitting.txt")
        args['synth_val_file'] = os.path.join(BASE_PATH, "BodyMAP/data_files/synth_overfitting.txt")
        args['real_train_file'] = os.path.join(BASE_PATH, "BodyMAP/data_files/real_overfitting.txt")
        args['real_val_file'] = os.path.join(BASE_PATH, "BodyMAP/data_files/real_overfitting.txt")
        args['epochs'] = args.get('epochs', 10)
    elif args['exp_type'] == 'normal' and args['exp_run'] == 'val': 
        # in this setting we use 1 to 70 for train and 70 to 80 for val (real), 1 to 70 for train and 70 to 80 for val (synth)
        args['synth_train_file'] = os.path.join(BASE_PATH, "BodyMAP/data_files/synth_train.txt")
        args['synth_val_file'] = os.path.join(BASE_PATH, "BodyMAP/data_files/synth_val.txt")
        args['real_train_file'] = os.path.join(BASE_PATH, "BodyMAP/data_files/real_train2.txt")
        args['real_val_file'] = os.path.join(BASE_PATH, "BodyMAP/data_files/real_val2.txt")
        args['epochs'] = args.get('epochs', 50)
    elif args['exp_type'] == 'normal' and args['exp_run'] == 'full-train-test':
        # in this setting we use 1 to 80 for train, 80 to 102 for val/test (real), all synth for train 
        args['synth_train_file'] = os.path.join(BASE_PATH, "BodyMAP/data_files/synth_all.txt")
        args['synth_val_file'] = os.path.join(BASE_PATH, "BodyMAP/data_files/synth_val.txt")
        args['real_train_file'] = os.path.join(BASE_PATH, "BodyMAP/data_files/real_train.txt")
        args['real_val_file'] = os.path.join(BASE_PATH, "BodyMAP/data_files/real_val.txt")
        args['epochs'] = args.get('epochs', 100)
    else:
        print ('ERROR: invalid setting')
    args['use_real'] = args.get('use_real', True)
    args['use_synth'] = args.get('use_synth', True)

    # data loading setup
    args['image_size_type'] = args.get('image_size_type', 'resized224')
    args['normalize_pressure'] = args.get('normalize_pressure', True)
    args['normalize_depth'] = args.get('normalize_depth', True)
    args['is_affine'] = args.get('is_affine', True)
    args['is_erase'] = args.get('is_erase', True)
    
    # model setup
    args['indexing_mode'] = args.get('indexing_mode', 9)
    args['main_model_fn'] = args.get('main_model_fn', 'PMM1')
    args['model_fn'] = args.get('model_fn', 'PME13')
    args['modality'] = args.get('modality', 'both')
    if args['image_size_type'] == 'original':
        args['feature_size_pressure'] = 16
        args['feature_size_depth'] = 64
        args['feature_size'] = MODALITY_TO_FEATURE_SIZE[args['modality']]
    else:
        args['feature_size_pressure'] = 392 # 392 for others
        args['feature_size_depth'] = 392 # 392 for others
        args['feature_size'] = 392
    args['vertex_size'] = 6890
    args['out_size'] = 88

    # opt setup
    args['lr'] = args.get('lr', 1e-4)
    args['weight_decay'] = args.get('weight_decay', 5e-4)
    args['batch_size'] = args.get('batch_size', 64)

    # loss setup
    args['contact_loss_fn'] = args.get('contact_loss_fn', 'ct0')
    args['v2v_loss'] = args.get('v2v_loss', False)
    args['pmap_loss'] = args.get('pmap_loss', True)
    args['smpl_loss'] = args.get('smpl_loss', True)

    # WS setup
    args['infer_smpl'] = args.get('infer_smpl', False)
    args['infer_pmap'] = args.get('infer_pmap', False)
    args['load_MOD1_path'] = args.get('load_MOD1_path', None)
    args['WS'] = args.get('WS', False)

    # loss weights setup
    args['pmap_loss_mult'] = args.get('pmap_loss_mult', 100)
    args['lambda_pmap_loss'] = args.get('lambda_pmap_loss', 0)
    args['lambda_contact_loss'] = args.get('lambda_contact_loss', 0)
    args['lambda_root_angle'] = args.get('lambda_root_angle', 1.0)
    args['lambda_smpl_loss'] = args.get('lambda_smpl_loss', 0.0)
    args['lambda_v2v_loss'] = args.get('lambda_v2v_loss', 0.0)
    args['lambda_proj_loss'] = args.get('lambda_proj_loss', 0.0)
    args['lambda_preg_loss'] = args.get('lambda_preg_loss', 0.0)

    # epochs settings setup
    args['epochs_metric'] = 25
    args['epochs_save'] = 10
    args['epochs_val_viz'] = args['epochs_metric'] 

    # exp setup
    args['exp str'] = args.get('exp str', 'exp str is not set')
    args['name'] = args.get('name', 'name not set')
    
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BodyMAP')
    parser.add_argument('args_file', type=str, 
                        help='Full path of json options file')
    args = parser.parse_args()
    exp = load_args_from_json(args.args_file)
    if exp is None:
        print ('Error in reading JSON file. Please check')
        exit(-1)
    
    args = parse_args(exp)

    if args.get('WS', False):
        trainer = WSTrainer(args)
    else:
        trainer = PMMTrainer(args)
    model = trainer.train_model()
    print ()

