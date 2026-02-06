# 四足机器人奖励函数设计与越障任务优化

## 01//INTRODUCTION 研究背景与目标

### 核心挑战
- 步态自然化：如何模拟生物动力学特性  
- 碰撞规避：复杂地形下的安全防护  
- 姿态稳定：凹凸地形中的平衡维持  

### 研究目标
- 融合运动控制(Locomotion)与导航任务(Navigation)
- 构建多维度稳定性保障体系  
- 突破波浪、楼梯、凹地形等越障难点  

---

## 02//GAIT OPTIMIZATION 步态自然化：空中滞留时间优化机制

### SWING TIME REWARD

#### REWARD FUNCTION MAPPING
  
区间范围：0.3s —— 0.5s（↑ MAX REWARD AT 0.5s, OPTIMAL）—— 0.7s

#### 触发机制
以“足端首次接触地面”为触发节点，计算空中滞留时间。

#### 奖励逻辑
- 基准奖励设于0.5s；
- 在0.3–0.7s区间内按线性比例调整权重。

#### 核心优势
- 模拟生物动力学特性  
- 提升凹凸地形跨越能力  
- 减少足端冲击载荷  

---

## 03//COLLISION DETECTION 碰撞检测：多维力传感器绊倒惩罚

### ALERT: IMPACT > THRESHOLD

#### DETECTION MODEL
$$
F = \sqrt{F_{x}^{2} + F_{y}^{2} + F_{z}^{2}} > 5 \times |F_{z}|
$$
即：总接触力 > 5 × 垂直方向力

#### 三维分量提取
实时监控垂直力 $ F_z $ 与水平切向力 $ F_x / F_y $ 的比例关系。

#### 状态判定
当水平冲击力显著大于垂直支撑力时，判定为“撞墙”或“绊倒”。

#### GRADIENT PENALTY MECHANISM

Initial penalty -1.0, accumulating -0.2 every 0.01s and capped at -5.0.

---

## 05//FLEXIBLE TRACKING 柔性追踪：Sigmoid距离奖励策略

### APPROACH REWARD FUNCTION

#### MATHEMATICAL MODEL
$$
R = \frac{1}{1 + e^{xy}}
$$
其中：
- $ x = 0.5 $：距离系数 (Distance Coefficient)  
- $ y = \text{当前距离} / \text{初始距离} $：距离比值 (Distance Ratio)

#### 平滑衰减 (Smooth Decay)
- 距离目标越近，奖励越接近1；远离时平滑衰减至0，无额外负惩罚。

#### 允许探索 (Exploration)
允许机器人在复杂环境中进行短暂探索或停留，避免陷入局部最优。

#### 降低僵硬 (Reduced Stiffness)
减少因过度距离约束导致的动作僵硬，提升路径规划的灵活性。

---

## 06//CHECKPOINT & STABILITY Checkpoint分段训练与姿态稳定性控制

### TRAINING STRATEGY

#### Checkpoint分段训练机制
- 采用多途径点（Waypoint）分段导航策略，降低长距离任务的训练难度。  
- 状态记忆：记录“最远达成 Checkpoint”，作为下一轮训练起点。  
- 奖励机制：到达奖励 = 基准值 (+5.0) + 序号递增 (每阶段+0.5)。

到达奖励递增序列如下表所示：

| 编号 | Checkpoint | 奖励值 |
|------|------------|--------|
| 1    | START → CP-1 | +5.0   |
| 2    | CP-1 → CP-2 | +5.5   |
| 3    | CP-2 → GOAL | +6.0   |

#### 姿态稳定性侧翻惩罚

##### 触发阈值
$$
|\text{Roll}| > 60^\circ \quad \text{或} \quad |\text{Pitch}| > 60^\circ
$$

##### 惩罚措施
- 给予强惩罚 (-10.0)  
- 立即终止当前训练回合  

##### 安全角度区间
- ROLL LIMIT: [-60°, 60°]
- PITCH LIMIT: [-60°, 60°]

---

## 07//PENALTY MECHANISM 停滞行为渐进式惩罚机制

### ANTI-STAGNATION LOGIC

#### 定义：停滞状态判定
机器人质心位移 < 0.1m / 10个控制周期

#### 渐进式惩罚机制

停滞时间步数越多，惩罚系数越大。

#### 设计理念
- **避免因短暂停滞被过度惩罚**，允许微调  
- **累计惩罚无上限**，迫使智能体寻找新路径  
- **配合柔性追踪奖励**，平衡探索与效率  

---

## 08//SYSTEM ARCHITECTURE 四足机器人稳定性控制方法体系

### MULTI-DIMENSIONAL STABILITY 多维稳定性框架

#### 1. 姿态稳定性控制 (Attitude)

CONSTRAINT: 约束姿态角在安全区间 (Roll/Pitch ∈ [-60°, 60°])
INPUT: 引入IMU角速度(ω)与角加速度(α)，感知姿态变化趋势
REWARD: 姿态平滑性奖励 (+0.1 / step)

#### 2. 步态动力学优化 (Gait Dynamics)

OPTIMIZATION: 优化空中滞留时间与接触力分布
CRITERIA: 符合ZMP（零力矩点）稳定判据
PROTECTION: 碰撞防护闭环控制

#### 3. 观测量维度扩展 (Observation)

STATE: 质心三维坐标/速度，关节角度/力矩
SENSOR: 足端接触力，IMU姿态数据
GOAL: 目标点相对位置

---

## 09//OBSTACLE BREAKTHROUGH Stage2越障任务：波浪地形与楼梯突破

### 波浪地形 (Wave Terrain)

#### CORE DIFFICULTIES
- × 姿态失稳：地表起伏引发俯仰角剧烈波动，侧翻风险高。  
- × 步幅不适：步幅与波长不匹配导致足端悬空或关节超限。  
- × 受力不均：地面法向量动态变化，接触力分布异常。  

#### BREAKTHROUGH STRATEGIES
- ✔ 初始稳定性强化：设置平地过渡段，完成步态初始化。  
- ✔ 自适应步幅调整：识别地形周期，动态调整空中停留时间基准。  
- ✔ 质心主动控制：增设质心高度稳定性奖励，维持平稳。  

---

### 楼梯地形 (Stairs Terrain)

#### CORE DIFFICULTIES
- × 参数通用性差：台阶尺寸与陡度差异大，步态难以自适应。  
- × 边缘打滑：足端落在台阶边缘易引发接触力突变与失衡。  
- × 动力学复杂：质心升降幅度大，受重力与惯性冲击影响显著。  

#### BREAKTHROUGH STRATEGIES
- ✔ 坡度自适应识别：建立“坡度-步态参数”映射，适配不同环境。  
- ✔ 足端定位优化：增加边缘距离奖励，引导足端落入中心区域。  
- ✔ 动力学补偿：根据力学特性调整奖励权重，适配升降运动。  

---

## 10∥CONCAVE TERRAIN Stage2越障任务：凹地形跨越策略

### GAP CROSSING LOGIC

#### 难点与解决方案对照表

| 编号 | 难点标识（Difficulty） | 具体问题描述                     | 解决方案标识（Solution） | 解决策略说明                                                                 |
|------|------------------------|----------------------------------|----------------------------|--------------------------------------------------------------------------------|
| 1    | [ERR_VAR_DIM]          | 凹陷尺寸比例差异                 | [STRATEGY_CLASS]             | 跨越策略分类奖励：根据凹陷宽度与腿长比例，设置不同奖励分支，引导选择适配通行方式。 |
| 2    | [ERR_CLIFF]            | 边缘悬崖效应                     | [SENS_EXTEND]             | 边缘探测辅助：利用传感器提前感知边缘，防止跌落。                               |
| 3    | [ERR_STUTTER]          | 过渡段步态切换卡顿               | [GAIT_SMOOTH]             | 步态切换平滑奖励：针对边缘步态切换动作设置奖励，引导实现平滑过渡，减少卡顿现象。 |

---

## 09//SUMMARY&OUTLOOK 总结与展望

### STATUS: VERIFIED

#### 技术突破 (TECHNICAL BREAKTHROUGHS)
- ✅ 多维度奖励机制融合：成功融合运动控制(Locomotion)与导航任务(Navigation)，实现步态自然化与柔性追踪。  
- ✅ 实时监测与防护体系：基于力传感器与IMU构建碰撞检测与姿态稳定性模型，形成闭环安全防护。  
- ✅ 高效分段训练策略：Checkpoint机制与渐进式惩罚协同，显著降低长距离任务训练难度。  

#### 未来展望 (FUTURE DIRECTIONS)
- 适配非结构化环境的搜救与巡检任务  
- 多模态传感器融合（视觉 + 激光雷达）  
- 端到端视觉导航与动态避障深度集成  
