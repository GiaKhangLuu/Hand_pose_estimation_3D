import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from datetime import datetime
import glob
from transformer_encoder import TransformerEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ann import ANN
from dataloader_ann import HandArmLandmarksDataset_ANN
import math
from sklearn.preprocessing import MinMaxScaler
from csv_writer import columns_to_normalize, fusion_csv_columns_name
import pandas as pd
import joblib
import numpy as np

from landmarks_scaler import LandmarksScaler

def fingers_length_loss(pred, gt):
    pred = pred.reshape(-1, 3, 21)
    gt = gt.reshape(-1, 3, 21)

    pred_wrist = pred[..., 0]  # assume that wrist is in the first position
    gt_wrist = gt[..., 0]  # (N, 3)

    pred_fingers = pred[..., 1:]  # (N, 3, 20)
    gt_fingers = gt[..., 1:]  # (N, 3, 20)

    pred_fingers = pred_fingers.reshape(-1, 3, 5, 4)  # (N, 3, 5, 4)
    gt_fingers = gt_fingers.reshape(-1, 3, 5, 4)  # (N, 3, 5, 4)

    pred_wrist = pred_wrist[..., None]  # (N, 3, 1)
    pred_wrist = pred_wrist.repeat_interleave(5, -1)  # (N, 3, 5)
    pred_wrist = pred_wrist[..., None]  # (N, 3, 5, 1)

    gt_wrist = gt_wrist[..., None]
    gt_wrist = gt_wrist.repeat_interleave(5, -1)
    gt_wrist = gt_wrist[..., None]

    pred_hand = torch.concatenate([pred_wrist, pred_fingers], axis=-1)  # (N, 3, 5, 5)
    gt_hand = torch.concatenate([gt_wrist, gt_fingers], axis=-1)  # (N, 3, 5, 5)

    pred_hand_distance = torch.zeros_like(pred_hand[..., :-1])  # (N, 3, 5, 4)
    gt_hand_distance = torch.zeros_like(gt_hand[..., :-1])  # (N, 3, 5, 4)

    for i in range(1, pred_hand_distance.shape[-1]):
        pred_hand_distance[..., i - 1] = pred_hand[..., i] - pred_hand[..., i - 1]
        gt_hand_distance[..., i - 1] = gt_hand[..., i] - gt_hand[..., i - 1]

    square_error = torch.square(pred_hand_distance - gt_hand_distance)  # (N, 3, 5, 4)

    mean_sum = torch.sqrt(torch.sum(square_error, axis=1))  # (N, 5, 4)

    loss = torch.mean(mean_sum)  # (1,)

    return loss

def landmark_distance_of_two_adjacent_fingers(pred, gt):
    pred = pred.reshape(-1, 3, 21)
    gt = gt.reshape(-1, 3, 21)

    pred_fingers = pred[..., 1:]  # (N, 3, 20)
    gt_fingers = gt[..., 1:]  # (N, 3, 20)

    pred_fingers = pred_fingers.reshape(-1, 3, 5, 4)  # (N, 3, 5, 4)
    gt_fingers = gt_fingers.reshape(-1, 3, 5, 4)  # (N, 3, 5, 4)

    pred_landmarks_distances = torch.zeros(pred_fingers.shape[0], 3, 4, 4)  # (N, 3, 4, 4)
    gt_landmarks_distances = torch.zeros(gt_fingers.shape[0], 3, 4, 4)  # (N, 3, 4, 4)

    for i in range(1, pred_fingers.shape[2]):
        pred_landmarks_distances[..., i - 1, :] = pred_fingers[..., i, :] - pred_fingers[..., i - 1, :]
        gt_landmarks_distances[..., i - 1, :] = gt_fingers[..., i, :] - gt_fingers[..., i - 1, :] 

    sq_error = torch.square(pred_landmarks_distances - gt_landmarks_distances)  # (N, 3, 4, 4)
    mean_sum = torch.sqrt(torch.sum(sq_error, axis=1))  # (N, 4, 4)
    loss = torch.mean(mean_sum)  # (1,)
    
    return loss

def fingers_length_loss_just_first_and_last_points(pred, gt):
    pred = pred.reshape(-1, 3, 21)
    gt = gt.reshape(-1, 3, 21)

    pred_wrist = pred[..., 0]  # assume that wrist is in the first position
    gt_wrist = gt[..., 0]  # (N, 3)

    pred_fingers = pred[..., 1:]  # (N, 3, 20)
    gt_fingers = gt[..., 1:]  # (N, 3, 20)

    pred_fingers = pred_fingers.reshape(-1, 3, 5, 4)  # (N, 3, 5, 4)
    gt_fingers = gt_fingers.reshape(-1, 3, 5, 4)  # (N, 3, 5, 4)

    pred_wrist = pred_wrist[..., None]  # (N, 3, 1)
    pred_wrist = pred_wrist.repeat_interleave(5, -1)  # (N, 3, 5)
    pred_wrist = pred_wrist[..., None]  # (N, 3, 5, 1)

    gt_wrist = gt_wrist[..., None]
    gt_wrist = gt_wrist.repeat_interleave(5, -1)
    gt_wrist = gt_wrist[..., None]

    pred_hand = torch.concatenate([pred_wrist, pred_fingers], axis=-1)  # (N, 3, 5, 5)
    gt_hand = torch.concatenate([gt_wrist, gt_fingers], axis=-1)  # (N, 3, 5, 5)

    pred_fingers_length = pred_hand[..., -1] - pred_hand[..., 0]  # (N, 3, 5)
    gt_fingers_length = gt_hand[..., -1] - gt_hand[..., 0]  # (N, 3, 5)

    square_error = torch.square(pred_fingers_length - gt_fingers_length)  # (N, 3, 5)

    mean_sum = torch.sqrt(torch.sum(square_error, axis=1)) # (N, 5)

    loss = torch.mean(mean_sum)  # (1,)

    return loss

def calculate_wrist_loss(pred, gt):
    pred = pred.reshape(-1, 3, 21)
    gt = gt.reshape(-1, 3, 21)

    # assume that wrist is in the first position
    pred_wrist = pred[..., 0]  # (N, 3)
    gt_wrist = gt[..., 0]  # (N, 3)

    sq_error = torch.square(pred_wrist - gt_wrist)  # (N, 3)
    mean_distance = torch.sqrt(torch.sum(sq_error, axis=1))

    loss = torch.mean(mean_distance)

    return loss

def calculate_index_finger_mcp_loss(pred, gt):
    pred = pred.reshape(-1, 3, 21)
    gt = gt.reshape(-1, 3, 21)

    # assume that wrist is in the first position
    pred_wrist = pred[..., 5]  # (N, 3)
    gt_wrist = gt[..., 5]  # (N, 3)

    sq_error = torch.square(pred_wrist - gt_wrist)  # (N, 3)
    mean_distance = torch.sqrt(torch.sum(sq_error, axis=1))

    loss = torch.mean(mean_distance)

    return loss

def calculate_middle_finger_mcp_loss(pred, gt):
    pred = pred.reshape(-1, 3, 21)
    gt = gt.reshape(-1, 3, 21)

    # assume that wrist is in the first position
    pred_wrist = pred[..., 9]  # (N, 3)
    gt_wrist = gt[..., 9]  # (N, 3)

    sq_error = torch.square(pred_wrist - gt_wrist)  # (N, 3)
    mean_distance = torch.sqrt(torch.sum(sq_error, axis=1))

    loss = torch.mean(mean_distance)

    return loss

def calculate_loss_between_thumb_pinky_and_index_ring_fingers(pred, gt):
    pred = pred.reshape(-1, 3, 21)
    gt = gt.reshape(-1, 3, 21)

    pred_fingers = pred[..., 1:]  # (N, 3, 20)
    gt_fingers = gt[..., 1:]  # (N, 3, 20)

    pred_fingers = pred_fingers.reshape(-1, 3, 5, 4)  # (N, 3, 5, 4)
    gt_fingers = gt_fingers.reshape(-1, 3, 5, 4)  # (N, 3, 5, 4)

    indexes = [(0, 4), (1, 3)]

    pred_landmarks_distances = torch.zeros(pred_fingers.shape[0], 3, 2, 4)  # (N, 3, 2, 4)
    gt_landmarks_distances = torch.zeros(gt_fingers.shape[0], 3, 2, 4)  # (N, 3, 2, 4)

    for i in range(len(indexes)):
        finger_idx = indexes[i]
        pred_landmarks_distances[..., i, :] = pred_fingers[..., finger_idx[1], :] - pred_fingers[..., finger_idx[0], :]
        gt_landmarks_distances[..., i, :] = gt_fingers[..., finger_idx[1], :] - gt_fingers[..., finger_idx[0], :] 

    sq_error = torch.square(pred_landmarks_distances - gt_landmarks_distances)  # (N, 3, 2, 4)
    mean_sum = torch.sqrt(torch.sum(sq_error, axis=1))  # (N, 2, 4)
    loss = torch.mean(mean_sum)  # (1,)
    
    return loss

def calculate_distance_of_each_landmark(pred, gt):
    pred = pred.reshape(-1, 3, 18)
    gt = gt.reshape(-1, 3, 18)

    sq_error = torch.square(pred - gt)  # (N, 3, 18)
    mean_sum = torch.sqrt(torch.sum(sq_error, axis=1)) # (N, 18)
    loss = torch.mean(mean_sum)  # (1,)

    return loss

def loss_function_with_anchors(pred, gt, hand_anchors):
    raw_pred = pred.clone()
    raw_gt = gt.clone()

    pred = pred.reshape(-1, 3, 18)
    gt = gt.reshape(-1, 3, 18)
    hand_anchors = hand_anchors.reshape(hand_anchors.shape[0], 3, -1)
    
    hand_anchor_indexes = [0, 5, 9]

    for i, joint_idx in enumerate(hand_anchor_indexes):
        batched_hand_anchor = hand_anchors[..., i]
        pred = torch.cat((pred[..., :joint_idx], batched_hand_anchor[..., None], pred[..., joint_idx:]), dim=-1)
        gt = torch.cat((gt[..., :joint_idx], batched_hand_anchor[..., None], gt[..., joint_idx:]), dim=-1) 

    pred = pred.reshape(pred.shape[0], -1)
    gt = gt.reshape(gt.shape[0], -1)

    loss_each_landmark = fingers_length_loss(pred, gt)
    loss_each_finger = fingers_length_loss_just_first_and_last_points(pred, gt)
    loss_distance_of_landmarks_in_two_adj_fingers = landmark_distance_of_two_adjacent_fingers(pred, gt)
    loss_distance_of_thumb_pinky_and_index_ring = calculate_loss_between_thumb_pinky_and_index_ring_fingers(pred, gt)
    independent_loss = calculate_distance_of_each_landmark(raw_pred, raw_gt)

    loss = (loss_each_landmark + 
        loss_distance_of_landmarks_in_two_adj_fingers +
        independent_loss)

    return loss

def train_model(model, 
    train_dataloader, 
    val_dataloader, 
    optimizer, 
    num_epochs=20, 
    save_path="model.pth",
    scheduler=None,
    writer=None,
    log_seq=100):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_dataloader:
            inputs = inputs.reshape(inputs.shape[0], 3, -1)
            hand_anchors = inputs[..., -3:]  # (B, 3, 3)
            hand_anchors = hand_anchors.reshape(hand_anchors.shape[0], -1)
            hand_anchors = hand_anchors.to("cuda")

            inputs = inputs[..., :-3]
            inputs = inputs.reshape(inputs.shape[0], -1)
            inputs = inputs.to("cuda")

            targets = targets.to("cuda")
            outputs = model(inputs)  

            train_loss = loss_function_with_anchors(outputs, targets, hand_anchors)            

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            running_loss += train_loss.item() 

        epoch_loss = running_loss / len(train_dataloader)
        train_losses.append(epoch_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in val_dataloader:
                val_inputs = val_inputs.reshape(val_inputs.shape[0], 3, -1)
                val_hand_anchors = val_inputs[..., -3:]  # (B, 3, 3)
                val_hand_anchors = val_hand_anchors.reshape(val_hand_anchors.shape[0], -1)
                val_hand_anchors = val_hand_anchors.to("cuda")

                val_inputs = val_inputs[..., :-3]
                val_inputs = val_inputs.reshape(val_inputs.shape[0], -1)
                val_inputs = val_inputs.to("cuda")
                val_targets = val_targets.to("cuda")
                val_outputs = model(val_inputs)

                val_batch_elementwise_loss = loss_function_with_anchors(val_outputs, val_targets, val_hand_anchors)
                
                val_loss += val_batch_elementwise_loss.item()
        val_loss = val_loss / len(val_dataloader)
        val_losses.append(val_loss)

        if scheduler is not None:
            scheduler.step(val_loss)

        if writer is not None: 
            # Log losses and learning rate to TensorBoard
            writer.add_scalar('Loss/Train', epoch_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        if (epoch + 1) % log_seq == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')
            print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}')

        # Save model if validation loss decreases
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved with Validation Loss: {val_loss:.4f}")

    return train_losses, val_losses

if __name__ == "__main__":
    # ----------- INIT MODEL --------------------
    input_dim = 144 + 144
    output_dim = 144
    hidden_dim = int(144 + (144 / 2))
    num_hidden_layers = 8
    dropout_rate = 0.1

    model = ANN(input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        dropout_rate=dropout_rate
    )
    model = model.to("cuda")

    # -------------- TRAINING SETUP --------------
    MODEL_NAME = "ann"
    DATETIME = "{}".format(datetime.now().strftime("%Y%m%d-%H%M"))
    DATE = "{}".format(datetime.now().strftime("%Y%m%d"))
    BASE_DIR = "Hand_pose_estimation_3D/arm_and_hand/runs/{}".format(MODEL_NAME)
    SAVE_DIR = os.path.join(BASE_DIR, DATE, DATETIME)
    DATA_DIR = "data"  
    writer = SummaryWriter(log_dir=SAVE_DIR)

    SELECTED_DATE = "*"
    train_paths = glob.glob(os.path.join(DATA_DIR, "{}/{}/fine_landmarks_{}_*.csv".format(SELECTED_DATE, SELECTED_DATE, "train")))
    val_paths = glob.glob(os.path.join(DATA_DIR, "{}/{}/fine_landmarks_{}_*.csv".format(SELECTED_DATE, SELECTED_DATE, "val")))
    body_lines = [[0,2], [0, 3], [2, 4], [3, 4]]
    lefthand_lines = [[0, 1], [1, 5], [5, 6], [5, 10], [5, 22], [10, 14], [14, 18], [18, 22], 
        [6, 7], [7, 8], [8, 9], 
        [10, 11], [11, 12], [12, 13], 
        [14, 15], [15, 16], [16, 17], 
        [18, 19], [19, 20], [20, 21], 
        [22, 23], [23, 24], [24, 25]]
    train_body_distance_thres = 550
    train_leftarm_distance_thres = 550
    train_lefthand_distance_thres = 200
    val_body_distance_thres=450,
    val_leftarm_distance_thres=450,
    val_lefthand_distance_thres=150,

    # Load the true dataset to get the scaler then pass the scaler to the true and fake dataset
    minmax_scaler = MinMaxScaler()
    train_dataset = HandArmLandmarksDataset_ANN(train_paths, 
        body_lines, 
        lefthand_lines, 
        train_body_distance_thres, 
        train_leftarm_distance_thres, 
        train_lefthand_distance_thres,
        filter_outlier=True,
        only_keep_frames_contain_lefthand=True)
    data_features = pd.DataFrame(train_dataset._inputs, columns=fusion_csv_columns_name[1:323])
    minmax_scaler.fit_transform(data_features[columns_to_normalize].values)
    scaler_save_path = os.path.join(SAVE_DIR, "input_scaler.pkl")
    joblib.dump(minmax_scaler, scaler_save_path)

    #fake_train_paths = glob.glob(os.path.join(DATA_DIR, "fake_data", "train", "fake_*.csv"))
    #fake_val_paths = glob.glob(os.path.join(DATA_DIR, "fake_data", "val", "fake_*.csv"))
    fake_train_paths = []
    fake_val_paths = []

    if len(fake_train_paths) > 0:
        train_paths.extend(fake_train_paths)
    if len(fake_val_paths) > 0:
        val_paths.extend(fake_val_paths)

    scaler = LandmarksScaler(columns_to_scale=columns_to_normalize,
        scaler_path=scaler_save_path)
    train_dataset = HandArmLandmarksDataset_ANN(train_paths, 
        body_lines, 
        lefthand_lines, 
        train_body_distance_thres, 
        train_leftarm_distance_thres, 
        train_lefthand_distance_thres,
        filter_outlier=True,
        only_keep_frames_contain_lefthand=True,
        scaler=scaler,
        keep_intrinsic_matrices=False)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_dataset = HandArmLandmarksDataset_ANN(val_paths,
        body_lines,
        lefthand_lines,
        val_body_distance_thres,
        val_leftarm_distance_thres,
        val_lefthand_distance_thres,
        filter_outlier=True,
        only_keep_frames_contain_lefthand=True,
        scaler=scaler,
        keep_intrinsic_matrices=False)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)  

    pretrained_weight_path = None
    if pretrained_weight_path is not None and os.path.exists(pretrained_weight_path):
        model.load_state_dict(torch.load(pretrained_weight_path))
        print("Loaded existing model weights: ", pretrained_weight_path)

    #criterion = nn.MSELoss(reduction="mean")
    criterion = nn.SmoothL1Loss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 50000
    current_time = datetime.now().strftime('%Y%m%d-%H%M')
    save_path = os.path.join(SAVE_DIR, "{}_{}_layers_best.pth".format(MODEL_NAME, num_hidden_layers))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', 
        factor=math.sqrt(0.1), patience=1000, verbose=True, min_lr=1e-8)
    #early_stopping = EarlyStopping(patience=7000, verbose=True)
    early_stopping = None

    train_losses, val_losses = train_model(model, 
        train_dataloader, 
        val_dataloader, 
        criterion, 
        optimizer, 
        num_epochs=num_epochs, 
        save_path=save_path,
        early_stopping=early_stopping,
        scheduler=scheduler,
        writer=writer,
        log_seq=50)

    writer.close()