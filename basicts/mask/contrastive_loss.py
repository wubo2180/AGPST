import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for self-supervised pre-training
    Helps learn better representations by contrasting positive/negative pairs
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z1, z2):
        """
        Args:
            z1, z2: (B, D) embeddings from two augmented views
        Returns:
            loss: contrastive loss value
        """
        B = z1.shape[0]
        
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Concatenate embeddings
        representations = torch.cat([z1, z2], dim=0)  # (2B, D)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature
        
        # Create labels: positives are (i, i+B) and (i+B, i)
        labels = torch.cat([torch.arange(B) + B, torch.arange(B)]).to(z1.device)
        
        # Mask out self-similarities
        mask = torch.eye(2 * B, dtype=torch.bool, device=z1.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
        
        # Compute contrastive loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


class TemporalContrastiveLoss(nn.Module):
    """
    Temporal contrastive loss for time series
    Contrasts adjacent vs. distant time segments
    """
    def __init__(self, temperature=0.5, negative_samples=5):
        super().__init__()
        self.temperature = temperature
        self.negative_samples = negative_samples
        
    def forward(self, embeddings, time_indices):
        """
        Args:
            embeddings: (B, N, L, D) patch embeddings
            time_indices: (B, L) time indices for each patch
        """
        B, N, L, D = embeddings.shape
        
        # Reshape: (B*N*L, D)
        emb_flat = embeddings.reshape(-1, D)
        emb_flat = F.normalize(emb_flat, dim=1)
        
        # Sample anchor, positive (adjacent), negative (distant) pairs
        # For simplicity, use consecutive patches as positives
        anchors = emb_flat[:-1]
        positives = emb_flat[1:]
        
        # Sample random negatives
        neg_idx = torch.randint(0, emb_flat.shape[0], (anchors.shape[0], self.negative_samples))
        negatives = emb_flat[neg_idx]  # (B*N*L-1, K, D)
        
        # Compute similarities
        pos_sim = (anchors * positives).sum(dim=1, keepdim=True) / self.temperature  # (B*N*L-1, 1)
        neg_sim = torch.matmul(anchors.unsqueeze(1), negatives.transpose(1, 2)).squeeze(1) / self.temperature  # (B*N*L-1, K)
        
        # Contrastive loss: maximize pos_sim, minimize neg_sim
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # (B*N*L-1, K+1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss


class EnhancedPretrainingLoss(nn.Module):
    """
    Combined loss for better pre-training:
    1. Masked reconstruction (original)
    2. Contrastive learning (new)
    3. Temporal consistency (new)
    """
    def __init__(self, alpha=0.6, beta=0.2, gamma=0.2, temperature=0.5):
        super().__init__()
        self.alpha = alpha  # Weight for reconstruction
        self.beta = beta    # Weight for contrastive
        self.gamma = gamma  # Weight for temporal
        
        self.contrastive = ContrastiveLoss(temperature)
        self.temporal = TemporalContrastiveLoss(temperature)
        
    def forward(self, reconstruction, target, encoder_output, time_indices=None):
        """
        Args:
            reconstruction: Reconstructed masked patches
            target: Ground truth masked patches
            encoder_output: Encoder embeddings for contrastive learning
            time_indices: Time indices for temporal contrastive
        """
        # Reconstruction loss (MAE)
        recon_loss = F.l1_loss(reconstruction, target)
        
        total_loss = self.alpha * recon_loss
        
        # Contrastive loss (if encoder output provided)
        if encoder_output is not None and len(encoder_output.shape) == 4:
            B, N, L, D = encoder_output.shape
            
            # Create two views by random masking/augmentation
            # For simplicity, use different subsets of patches
            mid = L // 2
            z1 = encoder_output[:, :, :mid, :].mean(dim=[1, 2])  # (B, D)
            z2 = encoder_output[:, :, mid:, :].mean(dim=[1, 2])  # (B, D)
            
            contrast_loss = self.contrastive(z1, z2)
            total_loss = total_loss + self.beta * contrast_loss
        
        # Temporal contrastive loss
        if time_indices is not None and encoder_output is not None:
            temp_loss = self.temporal(encoder_output, time_indices)
            total_loss = total_loss + self.gamma * temp_loss
        
        return total_loss, {
            'recon_loss': recon_loss.item(),
            'contrast_loss': contrast_loss.item() if encoder_output is not None else 0,
            'temporal_loss': temp_loss.item() if time_indices is not None else 0
        }
