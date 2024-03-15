import torch
import torch.nn.functional as F
import einops

# def foot_contact_loss_(model_contact, model_xp):
#     foot_idx = [7, 8, 10, 11]
#     loss_fn =  F.l1_loss
#     static_idx = model_contact > 0.95  # N x S x 4
#     model_feet = model_xp[:, :, foot_idx]  # foot positions (N, S, 4, 3)
#     model_foot_v = torch.zeros_like(model_feet)
#     model_foot_v[:, :-1] = (
#         model_feet[:, 1:, :, :] - model_feet[:, :-1, :, :]
#     )  # (N, S-1, 4, 3)
#     model_foot_v[~static_idx] = 0               # 不计算动态帧
#     foot_loss = loss_fn(                   # 静态的foot，让它的速度为0
#         model_foot_v, torch.zeros_like(model_foot_v), reduction="none"
#     )
#     # print("debug! before reduce:", foot_loss.shape)
#     foot_loss = einops.reduce(foot_loss, "b ... -> b (...)", "mean")
#     # print("debug! after reduce:", foot_loss.shape)
    
#     return foot_loss

def foot_contact_loss_(model_xp, static_idx):
    foot_idx = [7, 8, 10, 11]
    loss_fn =  F.l1_loss
    # static_idx = model_contact > 0.95  # N x S x 4
    model_feet = model_xp[:, :, foot_idx]  # foot positions (N, S, 4, 3)
    model_foot_v = torch.zeros_like(model_feet)
    model_foot_v[:, :-1] = (
        model_feet[:, 1:, :, :] - model_feet[:, :-1, :, :]
    )  # (N, S-1, 4, 3)
    
    model_foot_v_norm = torch.linalg.norm(model_foot_v, axis=-1)  # [BatchSize, 4, Frames]
    if static_idx == None:
        static_idx = torch.unsqueeze((model_foot_v_norm <= 0.01), dim=-1).repeat(1, 1, 1, 3)
    # print("model_foot_v_norm.shape", model_foot_v_norm.shape)
    # print("static_idxshape",static_idx.shape)
    # print("model_foot_v",model_foot_v.shape)
    
    
    
    model_foot_v[~static_idx] = 0               # 不计算动态帧
    foot_loss = loss_fn(                   # 静态的foot，让它的速度为0
        model_foot_v, torch.zeros_like(model_foot_v), reduction="none"
    )
    foot_loss = einops.reduce(foot_loss, "b ... -> b (...)", "mean")
    
    return foot_loss
    
class Foot_contact_loss():
    def __init__(self):
        pass

    def __call__(self, q, p):
        div = foot_contact_loss_(q, p)
        return div.mean()

    def __repr__(self):
        return "foot_contact_loss()"
    
    
if __name__ == "__main__":    
    print("test")
    model_xp = torch.rand([10, 150, 55, 3])
    model_contact = torch.randint(0, 2, (10, 150, 4)) 
    
    foot_contact_loss_(model_contact, model_xp)