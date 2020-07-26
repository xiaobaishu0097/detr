import argparse
import torch
import pandas as pd
from pathlib import Path

def load_eval(eval_path):
    data = torch.load(eval_path)
    # precision is n_iou, n_points, n_cat, n_area, max_det
    precision = data['precision']
    # take precision for all classes, all areas and 100 detections
    CLASSES = [
        'AlarmClock', 'Book', 'Bowl', 'CellPhone', 'Chair', 'CoffeeMachine', 'DeskLamp', 'FloorLamp',
        'Fridge', 'GarbageCan', 'Kettle', 'Laptop', 'LightSwitch', 'Microwave', 'Pan', 'Plate', 'Pot',
        'RemoteControl', 'Sink', 'StoveBurner', 'Television', 'Toaster',
    ]
    CLASSES = [c for c in CLASSES if c != 'N/A']
    area = 0
    return pd.DataFrame.from_dict({c: p for c, p in zip(CLASSES, precision[0, :, :, area, -1].mean(0) * 100)}, orient='index')

parser = argparse.ArgumentParser()
parser.add_argument('--eval_path', type=str)
args = parser.parse_args()

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    df = load_eval(args.eval_path)
    print(df)