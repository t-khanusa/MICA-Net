import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cross_decomposition import CCA


class CCA_Model(nn.Module):
    def __init__(self, cca_hidden_dim=64, num_classes=32):  # Back to 64 dims
        super(CCA_Model, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Linear(768, cca_hidden_dim),  # Direct to 64 dims
            nn.BatchNorm1d(cca_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(cca_hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, cca_hidden_dim)  # Back to cca_hidden_dim for CCA loss
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(600, cca_hidden_dim),  # Direct to 64 dims
            nn.BatchNorm1d(cca_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(cca_hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, cca_hidden_dim)  # Back to cca_hidden_dim for CCA loss
        )

        self.classifier = nn.Sequential(
            nn.Linear(cca_hidden_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_inertial, x_video):
        h_inertial = self.encoder1(x_inertial)
        h_video = self.encoder2(x_video)

        h_combined = torch.cat((h_inertial, h_video), dim=1)

        output = self.classifier(h_combined)

        return h_inertial, h_video, output

class CCA_Loss(nn.Module):
    def __init__(self, out_dim=64, n_components=None):
        super(CCA_Loss, self).__init__()
        self.out_dim = out_dim
        self.n_components = n_components or min(out_dim, 32)  # Default to reasonable number

    def forward(self, h1, h2):
        batch_size = h1.size(0)

        n_components = min(self.n_components, batch_size - 1, self.out_dim)

        if n_components <= 0:
            h1_norm = F.normalize(h1, p=2, dim=1)
            h2_norm = F.normalize(h2, p=2, dim=1)
            correlation = torch.sum(h1_norm * h2_norm, dim=1).mean()
            return -correlation

        h1_np = h1.detach().cpu().numpy()
        h2_np = h2.detach().cpu().numpy()

        h1_np = h1_np - np.mean(h1_np, axis=0, keepdims=True)
        h2_np = h2_np - np.mean(h2_np, axis=0, keepdims=True)

        try:
            # Initialize CCA
            cca = CCA(n_components=n_components, max_iter=500, tol=1e-6)

            # Fit CCA and transform
            h1_c, h2_c = cca.fit_transform(h1_np, h2_np)

            # Compute canonical correlations
            canonical_correlations = []
            for i in range(n_components):
                corr = np.corrcoef(h1_c[:, i], h2_c[:, i])[0, 1]
                if not np.isnan(corr):
                    canonical_correlations.append(abs(corr))  # Take absolute value

            if len(canonical_correlations) == 0:
                # Fallback if no valid correlations
                h1_norm = F.normalize(h1, p=2, dim=1)
                h2_norm = F.normalize(h2, p=2, dim=1)
                correlation = torch.sum(h1_norm * h2_norm, dim=1).mean()
                return -correlation

            # Sum of canonical correlations (we want to maximize this)
            total_canonical_correlation = sum(canonical_correlations)

            # Get the canonical weights/loadings for creating differentiable loss
            # cca.x_weights_ and cca.y_weights_ contain the transformation matrices
            W1 = torch.tensor(cca.x_weights_, device=h1.device, dtype=h1.dtype)
            W2 = torch.tensor(cca.y_weights_, device=h1.device, dtype=h1.dtype)

            # Apply the learned canonical transformations to get canonical variates
            h1_canonical = h1 @ W1  # (batch_size, n_components)
            h2_canonical = h2 @ W2  # (batch_size, n_components)

            # Compute differentiable canonical correlations
            # Normalize canonical variates
            h1_canonical = F.normalize(h1_canonical, p=2, dim=0)  # Normalize across samples
            h2_canonical = F.normalize(h2_canonical, p=2, dim=0)  # Normalize across samples

            # Compute correlations between corresponding canonical variates
            canonical_correlations_torch = []
            for i in range(n_components):
                # Pearson correlation coefficient between canonical variates
                h1_i = h1_canonical[:, i]
                h2_i = h2_canonical[:, i]

                # Center
                h1_i_centered = h1_i - h1_i.mean()
                h2_i_centered = h2_i - h2_i.mean()

                # Compute correlation
                numerator = (h1_i_centered * h2_i_centered).sum()
                denominator = torch.sqrt((h1_i_centered ** 2).sum() * (h2_i_centered ** 2).sum())

                if denominator > 1e-8:  # Avoid division by zero
                    corr = numerator / denominator
                    canonical_correlations_torch.append(torch.abs(corr))

            if len(canonical_correlations_torch) > 0:
                # Sum of canonical correlations (differentiable)
                total_canonical_corr = torch.stack(canonical_correlations_torch).sum()
                return -total_canonical_corr  # Negative to minimize
            else:
                # Fallback
                h1_norm = F.normalize(h1, p=2, dim=1)
                h2_norm = F.normalize(h2, p=2, dim=1)
                correlation = torch.sum(h1_norm * h2_norm, dim=1).mean()
                return -correlation

        except Exception as e:
            h1_norm = F.normalize(h1, p=2, dim=1)
            h2_norm = F.normalize(h2, p=2, dim=1)
            correlation = torch.sum(h1_norm * h2_norm, dim=1).mean()
            return -correlation

def train_model_with_cca(model, dataloaders, cca_alpha, num_epochs=25):
    print("Start training!")
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Initialize loss functions
    cca_loss_fn = CCA_Loss(out_dim=64)  # Back to 64 dims
    classification_loss_fn = nn.CrossEntropyLoss()

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_cca_loss = 0.0
            running_class_loss = 0.0
            running_corrects = 0

            for input1, input2, labels in dataloaders[phase]:
                input1 = input1.to(device)
                input2 = input2.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Model output is now 3 tensors
                    h_inertial, h_video, logits = model(input1, input2)

                    # Calculate Classification Loss
                    class_loss = classification_loss_fn(logits, labels)

                    # Calculate CCA Loss
                    cca_loss = cca_loss_fn(h_inertial, h_video)

                    # Combine losses
                    loss = class_loss + cca_alpha * cca_loss

                    # Get predictions
                    _, preds = torch.max(logits, 1)

                    # Only backward + optimize if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * input1.size(0)
                running_class_loss += class_loss.item() * input1.size(0)
                running_cca_loss += cca_loss.item() * input1.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_class_loss = running_class_loss / len(dataloaders[phase].dataset)
            epoch_cca_loss = running_cca_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Total Loss: {:.4f}, Class Loss: {:.4f}, CCA Loss: {:.4f}, Acc: {:.4f}'.format(
                  phase, epoch_loss, epoch_class_loss, epoch_cca_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history



class EarlyFusionModel(nn.Module):
    def __init__(self, dim_i=768, dim_v=600, num_classes=32, hidden_dim=256):
        super(EarlyFusionModel, self).__init__()
        
        # Classifier network that takes the concatenated vector as input
        self.classifier = nn.Sequential(
            nn.Linear(dim_i + dim_v, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, num_classes)
            # ,
            # nn.ReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, F_i, F_v):
        F_fused = torch.cat((F_i, F_v), dim=1)
        
        logits = self.classifier(F_fused)
        
        return logits

class LateFusionModel(nn.Module):
    def __init__(self, dim_i=768, dim_v=600, num_classes=32, hidden_i=128, hidden_v=128):
        super(LateFusionModel, self).__init__()

        # Classifier for the inertial modality (F_i)
        self.classifier_i = nn.Sequential(
            nn.Linear(dim_i, hidden_i),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_i, num_classes)
        )
        
        # Classifier for the video modality (F_v)
        self.classifier_v = nn.Sequential(
            nn.Linear(dim_v, hidden_v),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_v, num_classes)
        )

    def forward(self, F_i, F_v):
        logits_i = self.classifier_i(F_i)
        logits_v = self.classifier_v(F_v)
        
        P_i = F.softmax(logits_i, dim=1)
        P_v = F.softmax(logits_v, dim=1)
        
        P_fused = (P_i + P_v) / 2.0
        
        return P_fused
