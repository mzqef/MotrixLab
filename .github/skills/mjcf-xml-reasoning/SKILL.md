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
