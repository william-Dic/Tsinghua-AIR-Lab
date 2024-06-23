## This is all the materials for Tsinghua-AIR-Lab

# Learning Path
- Contrastive Learning 
- CLIP
- DecisionNCE
- Diffusion model（DDPM）
- bearobot
- Diffusion for robot learning
- IVM （Instruction-guided Visual Masking） https://arxiv.org/html/2405.19783v1
  ![image](utils/IVM.png)

sam - 用来分割segment 当有masking（掩码）的时候 该区域为1剩下区域为0 因此比较pixel来做loss（BCEWithLogitsLoss） forward
LoRA（Low-Rank Adaptation） - W≈A×B 通过这两个低秩矩阵的乘积 我们可以得到一个近似的𝑊 通过低秩矩阵分解减少模型的参数数量和计算复杂度
discriminnator 判断huamn/generated label
- DWBC（Discriminator-Weighted Offline Imitation Learning） https://github.com/ryanxhr/DWBC https://arxiv.org/abs/2207.10050
- LCBC (Language-Conditioned Behavior Cloning)
