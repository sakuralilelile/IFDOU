import os
import argparse
import torch

from douzero.evaluation.simulation import evaluate


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                    'Dou Dizhu Evaluation')
    parser.add_argument('--gpu_device', type=str, default='0')

    landlord = 'save_pkt/ding_save/landlord_weights_96134400.ckpt'
    landlord_up = 'save_pkt/sl/landlord_up.ckpt'
    landlord_down = 'save_pkt/sl/landlord_down.ckpt'
    eval_data = 'eval_data.pkl'
    num_workers = 5

    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    landlord_wp, landlord_adp, farmer_wp, farmer_adp = evaluate(landlord,
                                                                landlord_up,
                                                                landlord_down,
                                                                eval_data,
                                                                num_workers)


