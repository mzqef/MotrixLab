---
name: mjcf-xml-reasoning
description: Master understanding and reasoning about MuJoCo MJCF XML model files. Enables accurate interpretation of robot/scene definitions, kinematic trees, physics configurations, and simulation parameters.
---

## Purpose

MJCF (MuJoCo XML Format) is the native modeling language for MuJoCo physics simulation. This skill enables:

- **Reading and understanding** robot and scene XML definitions
- **Reasoning about** kinematic structures, joint configurations, contact physics
- **Debugging** physics issues through XML inspection
- **Designing** new scenes, terrains, and robot modifications
- **Connecting** XML elements to simulation behavior

## MJCF Document Structure

Every MJCF file has a single root `<mujoco>` element with optional sections:

```xml
<mujoco model="model_name">
  <compiler .../>     <!-- Parser/compiler settings -->
  <option .../>       <!-- Physics simulation options -->
  <size .../>         <!-- Memory allocation hints -->
  <default .../>      <!-- Default attribute values (cascading) -->
  <asset .../>        <!-- Meshes, textures, materials, height fields -->
  <worldbody .../>    <!-- Kinematic tree (bodies, geoms, joints) -->
  <contact .../>      <!-- Contact pairs, exclusions -->
  <equality .../>     <!-- Equality constraints (welds, joints) -->
  <tendon .../>       <!-- Spatial/fixed tendons -->
  <actuator .../>     <!-- Motors, position servos, etc. -->
  <sensor .../>       <!-- Sensors (IMU, force, position) -->
  <keyframe .../>     <!-- Named poses/configurations -->
  <visual .../>       <!-- Rendering settings -->
</mujoco>
```

## Core Element Reference

### `<compiler>` - Parser Settings

Controls how the XML is parsed and compiled:

| Attribute | Description | Example |
|-----------|-------------|---------|
| `angle` | Angle units (`degree`/`radian`) | `angle="radian"` |
| `meshdir` | Directory for mesh files | `meshdir="assets"` |
| `texturedir` | Directory for textures | `texturedir="textures"` |
| `autolimits` | Infer joint limits from range | `autolimits="true"` |
| `inertiafromgeom` | Compute inertia from geoms | `inertiafromgeom="true"` |

```xml
<compiler angle="radian" meshdir="assets" texturedir="assets" autolimits="true"/>
```

### `<option>` - Simulation Settings

Physics engine configuration:

| Attribute | Description | Default |
|-----------|-------------|---------|
| `timestep` | Integration step (seconds) | 0.002 |
| `gravity` | Gravity vector | "0 0 -9.81" |
| `integrator` | `Euler`, `RK4`, `implicit` | Euler |
| `cone` | Friction cone (`pyramidal`/`elliptic`) | pyramidal |
| `impratio` | Friction impedance ratio | 1 |

```xml
<option gravity="0 0 -9.81" timestep="0.01" integrator="RK4" cone="elliptic" impratio="100"/>
```

### `<default>` - Cascading Defaults (CSS-like)

Defaults eliminate redundancy. Classes can nest and inherit:

```xml
<default class="main">
  <geom friction="0.8" contype="1" conaffinity="1"/>
  <joint damping="1" armature="0.01"/>
  
  <default class="visual">
    <geom contype="0" conaffinity="0" group="2"/>  <!-- No collision -->
  </default>
  
  <default class="collision">
    <geom group="3"/>  <!-- Collision geometry -->
    <default class="foot">
      <geom type="sphere" size="0.03" condim="6" friction="0.8 0.02 0.01"/>
    </default>
  </default>
</default>
```

**Usage:** Reference with `class="classname"`:
```xml
<geom name="visual_body" mesh="body" class="visual"/>
<geom name="contact_foot" class="foot"/>
```

### `<asset>` - Resources

Contains meshes, textures, materials, height fields:

```xml
<asset>
  <!-- Mesh files (OBJ, STL) -->
  <mesh name="body_mesh" file="body.obj" scale="1 1 1"/>
  
  <!-- Height field terrain -->
  <hfield name="terrain" file="heightmap.png" size="10 10 0.5 0"/>
  
  <!-- Textures -->
  <texture name="ground_tex" type="2d" file="ground.png"/>
  <texture name="sky" type="skybox" builtin="gradient" rgb1="0.4 0.4 0.4" rgb2="0 0 0"/>
  
  <!-- Materials -->
  <material name="ground_mat" texture="ground_tex" texrepeat="0.4 0.4"/>
  <material name="red" rgba="0.8 0.1 0.1 1" specular="0.3" shininess="0.5"/>
</asset>
```

### `<worldbody>` - Kinematic Tree

The heart of the model. Bodies nest to form parent-child relationships:

```xml
<worldbody>
  <!-- Static ground (child of world) -->
  <geom name="floor" type="plane" size="0 0 0.01" material="ground_mat"/>
  <light pos="0 0 3" dir="0 0 -1" directional="true"/>
  
  <!-- Robot root body -->
  <body name="base" pos="0 0 0.5">
    <freejoint name="root"/>  <!-- 6-DOF floating base -->
    <inertial mass="10" pos="0 0 0" diaginertia="0.5 0.5 0.3"/>
    <geom name="torso" type="box" size="0.2 0.15 0.1" class="collision"/>
    
    <!-- Child body: leg -->
    <body name="thigh" pos="0.2 0 -0.1">
      <joint name="hip" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      <geom name="thigh_geom" type="capsule" fromto="0 0 0  0 0 -0.3" size="0.04"/>
      
      <!-- Grandchild: shank -->
      <body name="shank" pos="0 0 -0.3">
        <joint name="knee" type="hinge" axis="0 1 0" range="-2.8 0"/>
        <geom name="shank_geom" type="capsule" fromto="0 0 0  0 0 -0.3" size="0.03"/>
        <geom name="foot" type="sphere" size="0.03" pos="0 0 -0.3" class="foot"/>
      </body>
    </body>
  </body>
</worldbody>
```

## Body Elements Deep Dive

### `<body>` - Rigid Body

| Attribute | Description |
|-----------|-------------|
| `name` | Unique identifier |
| `pos` | Position relative to parent (x y z) |
| `quat` | Orientation quaternion (w x y z) |
| `euler`, `axisangle`, `xyaxes`, `zaxis` | Alternative orientation specs |
| `childclass` | Default class for all children |

### `<joint>` - Degrees of Freedom

Joints connect a body to its parent. Multiple joints in one body create composite joints.

| Type | DOF | Description |
|------|-----|-------------|
| `free` | 6 | Floating base (translation + rotation) |
| `ball` | 3 | Ball-and-socket (3D rotation) |
| `hinge` | 1 | Rotation around axis |
| `slide` | 1 | Translation along axis |

```xml
<!-- Hinge joint with limits -->
<joint name="hip_pitch" type="hinge" axis="0 1 0" pos="0 0 0" 
       limited="true" range="-1.57 1.57" damping="1" armature="0.01"/>

<!-- Free joint for floating base -->
<freejoint name="root"/>   <!-- Equivalent to <joint type="free"/>, fully unconstrained 6-DOF -->
```

Key joint attributes:
| Attribute | Description |
|-----------|-------------|
| `axis` | Rotation/translation axis (normalized) |
| `pos` | Joint position in body frame |
| `range` | Limits (min max) |
| `damping` | Velocity damping coefficient |
| `armature` | Rotor inertia (stabilizes simulation) |
| `stiffness` | Spring stiffness toward qpos0 |
| `frictionloss` | Dry friction torque |

### `<geom>` - Collision & Visual Geometry

| Type | Description | Size Parameters |
|------|-------------|-----------------|
| `plane` | Infinite ground plane | half-sizes for rendering |
| `sphere` | Sphere | radius |
| `capsule` | Cylinder with hemispherical caps | radius, half-length |
| `cylinder` | Cylinder | radius, half-length |
| `box` | Rectangular box | x, y, z half-sizes |
| `ellipsoid` | Ellipsoid | x, y, z radii |
| `mesh` | Triangle mesh | from mesh asset |
| `hfield` | Height field terrain | from hfield asset |

```xml
<!-- Box collision geom -->
<geom name="torso" type="box" size="0.2 0.15 0.1" rgba="0.8 0.2 0.2 1"/>

<!-- Capsule leg segment using fromto -->
<geom name="thigh" type="capsule" fromto="0 0 0  0 0 -0.3" size="0.04"/>

<!-- Mesh visual (no collision) -->
<geom type="mesh" mesh="body_visual" contype="0" conaffinity="0" group="2"/>

<!-- Height field terrain -->
<geom name="terrain" type="hfield" hfield="terrain_hfield" pos="0 0 0"/>
```

**Contact attributes:**
| Attribute | Description |
|-----------|-------------|
| `contype` | Collision type bitmask |
| `conaffinity` | Collision affinity bitmask |
| `condim` | Contact DOF (1=frictionless, 3=sliding, 4=+torsional, 6=+rolling) |
| `friction` | Friction coefficients (sliding torsional rolling) |
| `solimp` | Solver impedance (softness) |
| `solref` | Solver reference (stiffness, damping) |
| `priority` | Higher wins in parameter conflicts |

**Contact occurs when:** `(contype1 & conaffinity2) || (contype2 & conaffinity1)`

### `<site>` - Reference Points

Sites are massless markers for sensors, actuators, and tendons:

```xml
<site name="imu_site" pos="0 0 0" size="0.01"/>
<site name="ee_site" pos="0 0 -0.3" size="0.02"/>  <!-- End effector -->
```

### `<inertial>` - Mass Properties

Explicit inertia (overrides geom-based inference):

```xml
<inertial mass="10" pos="0 0.01 0.02" 
          quat="1 0 0 0" diaginertia="0.5 0.4 0.3"/>
```

| Attribute | Description |
|-----------|-------------|
| `mass` | Body mass (kg) |
| `pos` | Center of mass position |
| `diaginertia` | Principal moments (Ixx Iyy Izz) |
| `fullinertia` | Full 6-element inertia (Ixx Iyy Izz Ixy Ixz Iyz) |

## Actuators

### Types

| Type | Description | Key Params |
|------|-------------|------------|
| `motor` | Direct torque/force | `gear` |
| `position` | Position servo (PD) | `kp`, `kv` |
| `velocity` | Velocity servo | `kv` |
| `intvelocity` | Integrated velocity | `kp`, `actrange` |
| `general` | Fully configurable | `gaintype`, `biastype` |

```xml
<actuator>
  <!-- Direct motor -->
  <motor name="hip_torque" joint="hip" gear="100" ctrlrange="-3 3"/>
  
  <!-- Position servo with PD gains -->
  <position name="hip_pos" joint="hip" kp="200" kv="10" 
            ctrlrange="-1.57 1.57" forcerange="-100 100"/>
  
  <!-- Using defaults class -->
  <position class="servo" joint="knee" name="knee_pos"/>
</actuator>
```

### Transmission Types

| Transmission | Description |
|--------------|-------------|
| `joint` | Direct joint actuation |
| `tendon` | Tendon-based transmission |
| `site` | Force/torque at site |
| `body` | Adhesion forces (gecko feet) |

## Sensors

```xml
<sensor>
  <!-- Joint state -->
  <jointpos name="hip_pos" joint="hip"/>
  <jointvel name="hip_vel" joint="hip"/>
  <actuatorfrc name="hip_force" actuator="hip_motor"/>
  
  <!-- Body state -->
  <framepos name="base_pos" objtype="body" objname="base"/>
  <framelinvel name="base_vel" objtype="body" objname="base"/>
  <framequat name="base_quat" objtype="body" objname="base"/>
  
  <!-- IMU at site -->
  <gyro name="imu_gyro" site="imu_site"/>
  <accelerometer name="imu_accel" site="imu_site"/>
  
  <!-- Contact/force -->
  <touch name="foot_touch" site="foot_site"/>
  <force name="ankle_force" site="ankle_site"/>
</sensor>
```

## Contact Control

### Exclude Collisions

```xml
<contact>
  <exclude body1="base" body2="thigh"/>  <!-- Prevent self-collision -->
  <exclude body1="thigh" body2="shank"/>
</contact>
```

### Explicit Contact Pairs

```xml
<contact>
  <pair geom1="foot" geom2="ground" condim="6" friction="1 0.05 0.01"/>
</contact>
```

## Equality Constraints

```xml
<equality>
  <!-- Weld two bodies together -->
  <weld body1="gripper" body2="object"/>
  
  <!-- Lock joint to fixed angle -->
  <joint joint1="finger" polycoef="0 1 0 0 0"/>
  
  <!-- Connect bodies with offset -->
  <connect body1="arm" body2="tool" anchor="0 0 0.1"/>
</equality>
```

## MotrixLab Project XML Patterns

### Scene XML Structure (Navigation Tasks)

```xml
<mujoco>
  <asset>
    <hfield name="hfield_terrain" file="assets/heightmap.png" size="5 1.5 0.3 0"/>
    <material name="CollisionBlue" rgba="0.2 0.4 1.0 0.6"/>
  </asset>

  <default>
    <default class="collision">
      <geom type="mesh" contype="1" conaffinity="1"/>
      <default class="boundary">
        <geom type="box" rgba="0.2 0.4 1.0 0.6"/>
      </default>
    </default>
  </default>

  <worldbody>
    <body name="ground_root">
      <!-- Boundary walls -->
      <body name="boundaries" pos="0 0 0">
        <geom name="wall_left" class="boundary" size="0.25 10 1" pos="-5 0 1"/>
        <geom name="wall_right" class="boundary" size="0.25 10 1" pos="5 0 1"/>
      </body>
      
      <!-- Terrain heightfield -->
      <body name="terrain">
        <geom type="hfield" hfield="hfield_terrain" pos="0 0 0" 
              contype="1" conaffinity="1" friction="0.8"/>
      </body>
    </body>
  </worldbody>
</mujoco>
```

### Robot XML Structure (ANYmal/VBot)

```xml
<mujoco model="quadruped">
  <compiler angle="radian" meshdir="assets" autolimits="true"/>
  <option cone="elliptic" impratio="100"/>

  <default class="robot">
    <joint damping="1" frictionloss="0.1"/>
    <default class="visual">
      <geom contype="0" conaffinity="0" group="2"/>
    </default>
    <default class="collision">
      <geom group="3"/>
    </default>
    <default class="foot">
      <geom type="sphere" size="0.03" condim="6" friction="0.8 0.02 0.01"/>
    </default>
    <default class="servo">
      <position kp="200" kv="1" ctrlrange="-6.28 6.28" forcerange="-140 140"/>
    </default>
  </default>

  <worldbody>
    <body name="base" pos="0 0 0.5" childclass="robot">
      <freejoint/>
      <site name="imu_site"/>
      <geom class="collision" type="box" size="0.3 0.1 0.1"/>
      
      <!-- Leg: LF (Left Front) -->
      <body name="LF_hip" pos="0.3 0.1 0">
        <joint name="LF_HAA" type="hinge" axis="1 0 0" range="-0.7 0.5"/>
        <body name="LF_thigh" pos="0.06 0.07 0">
          <joint name="LF_HFE" type="hinge" axis="0 1 0"/>
          <body name="LF_shank" pos="0.1 0 -0.28">
            <joint name="LF_KFE" type="hinge" axis="0 1 0"/>
            <geom name="LF_FOOT" class="foot" pos="0.01 0.08 -0.31"/>
          </body>
        </body>
      </body>
      <!-- ... RF, LH, RH legs similar ... -->
    </body>
  </worldbody>

  <sensor>
    <framelinvel name="base_linvel" objtype="body" objname="base"/>
    <gyro name="base_gyro" site="imu_site"/>
  </sensor>

  <actuator>
    <position class="servo" joint="LF_HAA" name="LF_HAA"/>
    <position class="servo" joint="LF_HFE" name="LF_HFE"/>
    <position class="servo" joint="LF_KFE" name="LF_KFE"/>
    <!-- ... 12 total for quadruped ... -->
  </actuator>
</mujoco>
```

## Common Analysis Questions

| Question | Where to Look |
|----------|---------------|
| Why doesn't robot move? | `<actuator>` - check joint names, ctrl ranges |
| Robot flies away? | `<option>` timestep too large; `<joint>` missing damping |
| No foot contact? | `<geom>` contype/conaffinity mismatch; condim too low |
| Wrong mass? | `<inertial>` explicit mass; `<geom>` density attribute |
| Visual mismatch? | Check `meshdir`, `texturedir` paths |
| Self-collision issues? | Add `<contact><exclude>` pairs |
| Slippery feet? | Increase `friction` on foot geoms |
| Wobbly joints? | Increase `damping`, `armature` on joints |
| Terrain not working? | Check `<hfield>` size param: (x_ext y_ext z_max z_min) |

## Height Field (hfield) Sizing

```xml
<hfield name="terrain" file="heightmap.png" size="5 1.5 0.3 0"/>
<!--                                           ↑     ↑   ↑   ↑
                                               |     |   |   z_min (usually 0)
                                               |     |   z_max (max height)
                                               |     y half-extent
                                               x half-extent
-->
```

The PNG grayscale maps 0→z_min, 255→z_max.

## Frame Orientation Specification

Multiple ways to specify rotation:

| Attribute | Format | Example |
|-----------|--------|---------|
| `quat` | w x y z | `quat="1 0 0 0"` (no rotation) |
| `euler` | rx ry rz | `euler="0 0 1.57"` |
| `axisangle` | ax ay az angle | `axisangle="0 0 1 90"` |
| `xyaxes` | x1 x2 x3 y1 y2 y3 | `xyaxes="1 0 0 0 1 0"` |
| `zaxis` | z1 z2 z3 | `zaxis="0 0 1"` |

Only use **one** orientation attribute per element.

## Debugging Workflow

1. **Check compiler errors** - Run `mj_loadXML` to get error line numbers
2. **Visualize collision** - Set `group="3"` on collision geoms, toggle in viewer
3. **Check contact** - Use `mjVIS_CONTACTPOINT` to see contact locations
4. **Verify inertia** - Use `mjVIS_INERTIA` to see equivalent inertia boxes
5. **Test actuators** - Apply small control inputs, verify expected motion
6. **Check sensors** - Print `mjData.sensordata` to verify readings

## Quick Reference Cards

### Joint Naming Convention (Quadrupeds)

| Pattern | Meaning |
|---------|---------|
| `LF_`, `RF_`, `LH_`, `RH_` | Left/Right Front/Hind |
| `_HAA` | Hip Abduction/Adduction |
| `_HFE` | Hip Flexion/Extension |
| `_KFE` | Knee Flexion/Extension |

### Contact condim Values

| condim | Friction Type |
|--------|---------------|
| 1 | Frictionless (normal only) |
| 3 | Sliding friction (tangent plane) |
| 4 | + Torsional (spin around normal) |
| 6 | + Rolling (tilt around tangents) |

### Geom Group Conventions

| Group | Purpose |
|-------|---------|
| 0 | Default |
| 1 | Primary collision |
| 2 | Visual only (high detail) |
| 3 | Collision only (simplified) |

## Common Fixes

| Symptom | Fix |
|---------|-----|
| Bodies explode at start | Add `armature` to joints; reduce `timestep` |
| Robot sinks through floor | Check `contype`/`conaffinity` matching |
| Jerky contact | Increase `condim`; adjust `solref`/`solimp` |
| Mesh not found | Check `meshdir` path in `<compiler>` |
| Limits not working | Set `limited="true"` or `autolimits="true"` |
| Actuator does nothing | Verify joint name matches; check `ctrlrange` |

## MotrixLab XML Files

| Path | Description |
|------|-------------|
| `starter_kit/navigation1/vbot/xmls/*.xml` | Navigation1 terrain scenes (flat) |
| `starter_kit/navigation2/vbot/xmls/*.xml` | Navigation2 terrain scenes (obstacles/stairs) |
| `starter_kit/navigation1/anymal_c/xmls/anymal_c.xml` | ANYmal robot model |
| `motrix_envs/src/motrix_envs/basic/*/` | Basic env XMLs (cartpole, cheetah, etc.) |
| `motrix_envs/src/motrix_envs/common/` | Shared materials, skybox, visual configs |

## Description in Chinese



   

Searched for files matching `**/*.xml`, 23 matches

Read skill [mjcf-xml-reasoning](file:///d%3A/MotrixLab/.github/skills/mjcf-xml-reasoning/SKILL.md?vscodeLinkType%3Dskill)

Let me find and read the actual XML scene files. 

Searched for files matching `**/starter_kit/**/xmls/*.xml`, no matches

Read [](file:///d%3A/MotrixLab/starter_kit/navigation2/vbot/xmls)

Read [](file:///d%3A/MotrixLab/starter_kit/navigation1/vbot)

Created 6 todos

Read [](file:///d%3A/MotrixLab/starter_kit/navigation2/vbot/xmls/scene_section001.xml)

Read [](file:///d%3A/MotrixLab/starter_kit/navigation2/vbot/xmls/scene_section01.xml)

Read [](file:///d%3A/MotrixLab/starter_kit/navigation2/vbot/xmls/scene_section02.xml)

Read [](file:///d%3A/MotrixLab/starter_kit/navigation2/vbot/xmls/scene_section03.xml)

Starting: *Read collision model XMLs* (2/3)

Now let me read the collision model XML files:    

Read [](file:///d%3A/MotrixLab/starter_kit/navigation2/vbot/xmls/0131_C_section01.xml)

Read [](file:///d%3A/MotrixLab/starter_kit/navigation2/vbot/xmls/0126_C_section01.xml)

Read [](file:///d%3A/MotrixLab/starter_kit/navigation2/vbot/xmls/0126_C_section02.xml)

Read [](file:///d%3A/MotrixLab/starter_kit/navigation2/vbot/xmls/0126_C_section03.xml)

Now let me also check the environment configs to understand spawn positions and goals: 

Read [](file:///d%3A/MotrixLab/starter_kit/navigation2/vbot/cfg.py)

Starting: *Summarize all environments* (3/3)

Completed: *Summarize all environments* (3/3)

Here is the detailed analysis of all four environments. Coordinates are in meters; `size` for boxes = half-extents, for cylinders/spheres = radius.

---

## Section 001 — 平坦大圆台（训练热身关卡）

**碰撞模型**: 0131_C_section01.xml

| 要素 | 类型 | 中心坐标 (x, y, z) | 尺寸参数 | 顶面高度 z |
|------|------|---------------------|----------|-----------|
| 大圆柱台 (Cylinder01) | cylinder | (0, 0, **-1.0**) | 半径=**12.5m**, 半高=**1.0m** | **z = 0** |

**环境描述**:
- 一个巨大的平坦圆柱形平台，半径 12.5m，顶面在 z=0。
- **无任何障碍物、楼梯或坡道**，纯粹的平地导航训练。
- 机器人起点: (0, -2.4, 0.5)，目标: (0, 10.2, ?)，距离约 **12.6m**。
- 用途：学习基本前进和导航，为后续关卡暖身。

---

## Section 01 — 高度场 + 斜坡 + 高台

**碰撞模型**: 0126_C_section01.xml

### 地形高度场 (hfield)
| 要素 | 位置 (x, y, z) | 范围 | 高度 |
|------|----------------|------|------|
| hfield_terrain | (0, 0, 0) | x=±5m, y=±1.5m | 0 ~ 0.277m |

### 地面平台 (Adiban)
| 名称 | 中心坐标 (x, y, z) | 半尺寸 (x, y, z) | 顶面 z | 说明 |
|------|---------------------|-------------------|--------|------|
| Adiban_005 | (0, 0, -0.25) | 5.0×1.5×0.25 | **0** | 中央平台 |
| Adiban_001 | (0, -2.5, -0.25) | 5.0×1.0×0.25 | **0** | 后方平台（起跑区） |
| Adiban_002 | (0, 1.75, -0.25) | 5.0×0.25×0.25 | **0** | 过渡薄平台 |
| Adiban_003 | (0, **4.48**, **0.41**) | 5.0×2.5×0.25 | 倾斜面 | **斜坡**，绕 x 轴倾斜约 **15°** |
| **Adiban_004（高台）** | (0, **7.83**, **1.04**) | 5.0×1.0×0.25 | **1.294** | **主高台** |

### 边界墙 (Abianjie)
| 名称 | 中心坐标 (x, y, z) | 半尺寸 | 顶面 z |
|------|---------------------|--------|--------|
| 左墙 001 | (5.25, -1.0, 1.23) | 0.25×3.0×1.23 | ~2.45 |
| 左墙 002 | (5.25, 5.41, 1.23) | 0.25×3.41×1.25 | ~2.48 |
| 右墙 003 | (-5.25, -1.0, 1.23) | 0.25×3.0×1.23 | ~2.45 |
| 右墙 004 | (-5.25, 5.41, 1.23) | 0.25×3.41×1.23 | ~2.45 |
| 后墙 005 | (0, -3.75, 1.23) | 5.0×0.25×1.23 | ~2.45 |

**环境描述**:
- 机器人从 y≈-2.4 的平地出发 (z=0)，沿 +y 方向前进。
- 经过平台 → 高度场微地形 → **15° 斜坡**上行 → 到达**高台 (z≈1.294)**。
- 总行程约 **12.6m**（y: -2.4 → 10.2）。
- 挑战：上坡行走 + 高台攀登。

---

## Section 02 — 楼梯 + 球形障碍 + 吊桥 + 锥体

**碰撞模型**: 0126_C_section02.xml

### 地形高度场 (hfield)
| 要素 | 位置 (x, y, z) | 范围 | 高度 |
|------|----------------|------|------|
| hfield_terrain | (0, **10.33**, **1.294**) | x=±5m, y=±1.5m | 0 ~ 0.1m |

### 平台 (Bdiban)
| 名称 | 中心坐标 (x, y, z) | 半尺寸 | 顶面 z | 说明 |
|------|---------------------|--------|--------|------|
| Bdiban_005 | (0, 10.33, 1.044) | 5.0×1.5×0.25 | **1.294** | 起始平台（与 Section01 高台衔接） |
| Bdiban_001 | (0, 12.08, 1.044) | 5.0×0.25×0.25 | **1.294** | 过渡薄平台 |
| Bdiban_002 | (2.0, 14.45, 2.194) | 3.0×0.125×0.1 | **2.294** | 右侧楼梯顶部小平台 |
| Bdiban_003 | (2.0, 21.20, 2.194) | 3.0×0.125×0.1 | **2.294** | 右侧楼梯底部小平台 |
| Bdiban_004 | (0, 24.33, 1.194) | 5.0×1.0×0.1 | **1.294** | 终点平台 |
| **Adiban_004**（从 s01 继承） | (0, 7.83, 1.044) | 5.0×1.0×0.25 | **1.294** | 高台（仅此 geom 被启用） |

### 左侧楼梯（宽 4m，10 阶）
| 楼梯组 | x 中心 | y 范围 | z 范围（顶面） | 每阶ΔZ | 方向 |
|--------|--------|--------|---------------|--------|------|
| **left_stair1（上行）** | -3.0 | 12.43 → 14.23 | 1.369 → **2.794** | ~**0.15** | ↑ 上升 |
| **left_stair2（下行）** | -3.0 | 21.43 → 23.23 | **2.794** → 1.369 | ~0.15 | ↓ 下降 |
| **stair_diji_001** 斜面底板 | -3.0 | 13.48 (center) | 1.844 (center) | 斜面 | 上行底板 |
| **stair_diji_002** 斜面底板 | -3.0 | 22.18 (center) | 1.844 (center) | 斜面 | 下行底板 |

### 右侧楼梯（宽 6m，10 阶，更缓）
| 楼梯组 | x 中心 | y 范围 | z 范围（顶面） | 每阶ΔZ | 方向 |
|--------|--------|--------|---------------|--------|------|
| **right_stair1（上行）** | 2.0 | 12.43 → 14.23 | 1.319 → **2.294** | ~**0.10** | ↑ 上升 |
| **right_stair2（下行）** | 2.0 | 21.43 → 23.23 | **2.294** → 1.319 | ~0.10 | ↓ 下降 |

### 吊桥区（左侧 x≈-3.0）
| 要素 | 中心坐标 (x, y, z) | 尺寸 | 说明 |
|------|---------------------|------|------|
| 桥支撑圆柱 ×4 | (-1.4/-4.6, 14.93/20.73, 2.294) | R=0.4, 半高=0.5 | 桥墩 |
| 桥底座平台 ×4 | (-3.0, 14.6~21.0, 2.294) | box | 桥入口/出口平台 |
| 桥面台阶 ×23 | (-3.0, 15.31→20.33, 2.51→2.86→2.51) | ~2.64m宽 × 薄板 | **拱形吊桥**——中间凸起 |
| 桥栏杆 ×2 | (-1.68/-4.32, 17.83, 2.857) | 薄板 0.03m厚 | 两侧护栏 |
| 桥柱 ×4 | (-1.68/-4.32, 15.20/20.44, 2.141) | R=0.2, 半高=1.1 | 桥柱支撑 |

### 球形障碍 (Bpo_sphere) — 右侧通道
| 编号 | 中心坐标 (x, y, z) | 半径 | 顶面 z |
|------|---------------------|------|--------|
| sphere_001 | (3.50, 15.84, 1.20) | 0.75 | 1.95 |
| sphere_002 | (0.36, 15.84, 1.20) | 0.75 | 1.95 |
| sphere_003 | (2.00, 17.83, 0.80) | 0.75 | 1.55 |
| sphere_004 | (3.50, 19.72, 1.20) | 0.75 | 1.95 |
| sphere_005 | (0.36, 19.72, 1.20) | 0.75 | 1.95 |

### 其他障碍
| 类型 | 数量 | 说明 |
|------|------|------|
| 锥体 (cone_mesh) | 8 个 | STL 网格，位置由网格原点决定 |
| Logo 障碍 | 2 个 | STL 网格 |
| Bpo 主体 | 1 个 | STL 网格障碍 |

**环境描述**:
- 最复杂的关卡。从 Section01 高台 (z=1.294) 出发。
- **两条路线**：
  - **左路**：上 10 阶陡楼梯 (ΔZ=0.15/阶) → 走**拱形吊桥** → 下 10 阶楼梯
  - **右路**：上 10 阶缓楼梯 (ΔZ=0.10/阶) → 穿过**5 个球形障碍**和**锥体**密布的通道 → 下 10 阶楼梯
- 终点平台在 y≈24.33, z≈1.294。
- 机器人起点: (-2.5, 12.0, 1.8)。

---

## Section 03 — 金球障碍 + 斜坡 + 高台

**碰撞模型**: 0126_C_section03.xml

### 地形高度场 (hfield)
| 要素 | 位置 (x, y, z) | 范围 | 高度 |
|------|----------------|------|------|
| hfield_terrain | (0, **29.33**, **1.343**) | x=±5m, y=±1.5m | 0 ~ 0.149m |

### 平台与斜坡 (Cdiban)
| 名称 | 中心坐标 (x, y, z) | 半尺寸 | 顶面 z | 说明 |
|------|---------------------|--------|--------|------|
| Cdiban_001 | (0, 26.33, 1.044) | 5.0×1.0×0.25 | **1.294** | 入口平台（衔接 S02） |
| **Cdiban_006（竖向挡条）** | (0, 27.58, **0.544**) | 5.0×0.25×**0.75** | **1.294** | 台阶状高挡条 |
| **Cdiban_002（斜坡）** | (0, 27.62, 1.301) | 5.0×0.269×0.1 | 倾斜面 | 绕 x 轴倾斜约 **21.8°** |
| Cdiban_005 | (0, 29.33, 0.794) | 5.0×1.5×**0.5** | **1.294** | 中间厚平台 |
| **Cdiban_004** | (0, **32.33**, 0.994) | 5.0×1.5×**0.5** | **1.494** | **终点高台**（最高点） |

### 金球障碍 (goldball)
| 编号 | 中心坐标 (x, y, z) | 半径 | 顶面 z |
|------|---------------------|------|--------|
| goldball_001 | (**3.0**, 31.23, 0.844) | 0.75 | 1.594 |
| goldball_002 | (**0.0**, 31.23, 0.844) | 0.75 | 1.594 |
| goldball_003 | (**-3.0**, 31.23, 0.844) | 0.75 | 1.594 |

### 边界围挡 (Ccube)
| 名称 | 中心坐标 (x, y, z) | 半尺寸 | 顶面 z | 说明 |
|------|---------------------|--------|--------|------|
| Ccube_001 | (5.96, 27.84, 2.064) | 1.0×2.52×1.27 | **3.334** | 右墙 |
| Ccube_002 | (-5.96, 27.84, 2.064) | 1.0×2.52×1.27 | **3.334** | 左墙 |
| Ccube_003 | (5.52, 32.09, 2.064) | 1.0×1.73×1.27 | **3.334** | 右墙（末段） |
| Ccube_004 | (-5.52, 32.09, 2.064) | 1.0×1.73×1.27 | **3.334** | 左墙（末段） |
| **Ccube_005** | (-0.48, **34.33**, **2.564**) | 5.0×0.5×1.27 | **3.834** | **终点封堵墙** |

**环境描述**:
- 从 Section02 终点 (z=1.294) 进入。
- 经过**入口平台** → 翻越**0.75m 高挡条** → 过 **21.8° 陡斜坡** → 穿过中间平台。
- 在 y≈31.23 处有 **3 个金球 (R=0.75)** 横排挡路，间距 3m，需要绕行。
- 最终到达 **终点高台 (z=1.494)**。
- 终点后有封堵墙 (y=34.33)。
- 机器人起点: (0, 26.0, 1.8)。

---

## 全局纵览（赛道 Y 轴排布）

```
Y轴 →
-3.5   0    4.5   7.8   10.3  12.4  14.2  15~20  21.4  23.2  24.3  26.3  27.6  29.3  31.2  32.3  34.3
  |----S001/S01----|----S01高台----|-------Section 02 楼梯+吊桥+障碍------------|----Section 03 斜坡+金球----|
  起跑   坡道   高台z=1.29        楼梯上          吊桥/球区        楼梯下       挡条  斜坡  金球  终点z=1.49  墙
  z=0   z=0~1.29                 z=1.29→2.79    z=2.5~2.86      z=2.79→1.29       z=1.29→1.49
```

| Section | Y 范围 | Z 范围 | 核心难点 |
|---------|--------|--------|----------|
| **001** | -3.5 ~ 12.5 | 0 | 无障碍平地 |
| **01** | -3.5 ~ 8.8 | 0 → 1.294 | 15° 斜坡 + 高台攀登 |
| **02** | 8.8 ~ 24.3 | 1.294 → 2.794 → 1.294 | 楼梯(左陡/右缓) + 拱桥 + 球形障碍 + 锥体 |
| **03** | 24.3 ~ 34.3 | 1.294 → 1.494 | 0.75m 高挡条 + 21.8° 陡坡 + 3 金球 |