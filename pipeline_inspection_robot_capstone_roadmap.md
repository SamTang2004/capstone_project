# FPGA-Accelerated Pipeline Inspection Robot  
## Capstone Project Definition, System Architecture, Failure Modes, and 8-Month Roadmap

**Team composition:** 2 Computer Engineering students, 4 Electrical Engineering students  
**Target domain:** Underground stormwater / large-diameter pipe inspection  
**Primary objective:** Remotely operated pipe inspection robot with adaptive strut preload, multi-camera vision, distributed lighting, and FPGA-accelerated computer vision packaged as a black-box accelerator.

---

## 1. Executive Summary

This project proposes a remotely operated robot for inspecting large underground stormwater pipes. The robot uses eight spring-loaded struts to maintain pipe-wall contact, normal rubber wheels for locomotion, a centralized cable-preload mechanism to reduce wheel normal force during tank turns, and multiple cameras with distributed illumination for visual inspection.

The computer vision system uses two processing paths:

1. **Raspberry Pi / embedded Linux baseline CV**
2. **FPGA-accelerated CV black box with a defined API**

The FPGA is intentionally separated from the main robot control stack. The robot should remain drivable and recoverable even if the FPGA fails.

The final project goal is not full sewer-grade autonomous deployment. The realistic capstone goal is a controlled demonstration in a pipe test rig showing:

- forward/reverse movement,
- adaptive strut preload,
- tank-turn capability,
- T-junction or partial T-junction navigation,
- camera-based inspection,
- lighting control,
- Raspberry Pi CV baseline,
- FPGA-accelerated CV comparison,
- logged performance and power metrics.

---

## 2. Problem Statement

Underground stormwater and large utility pipes are difficult to inspect because they are long, dark, confined, wet, and may contain cracks, joint separation, corrosion, sediment, and blockages. Manual inspection is difficult, slow, and potentially unsafe.

A compact robotic inspection platform can reduce inspection difficulty by carrying cameras, lights, and onboard computer vision into the pipe while remaining remotely controlled through a tether.

---

## 3. Project Objectives

### 3.1 Core Objectives

1. Design and prototype a pipe-crawling robot for large circular pipes.
2. Maintain pipe-wall contact using spring-loaded struts.
3. Use a centralized cable mechanism to reduce strut preload during turns.
4. Remotely control the robot through a tether.
5. Capture live video using front and rear cameras.
6. Use distributed LED lighting to reduce strut shadows and improve CV quality.
7. Implement Raspberry Pi / embedded Linux baseline computer vision.
8. Implement FPGA-accelerated image processing as a black-box module with an API.
9. Compare FPGA and Raspberry Pi CV performance and power consumption.
10. Demonstrate the system in a controlled pipe test rig.

### 3.2 Stretch Objectives

1. Navigate a full T-junction.
2. Navigate a Y-junction.
3. Add side cameras.
4. Add semi-autonomous turn assist.
5. Add crack localization and inspection report generation.
6. Add more advanced FPGA-specific model acceleration.

---

## 4. Baseline System Concept

The robot consists of:

- central chassis,
- 8 spring-loaded struts,
- 4 driven wheels,
- 4 passive support wheels,
- centralized cable-opposed preload mechanism,
- front and rear cameras,
- distributed LED lighting,
- onboard Raspberry Pi-class computer,
- real-time microcontroller,
- FPGA CV accelerator,
- tethered communication and/or power.

### 4.1 High-Level Architecture

```text
Operator Laptop / Control Station
        │
        │ Ethernet / tether
        │
Raspberry Pi / Embedded Linux Supervisor
        │
        ├── Microcontroller: motors, preload, lighting, sensors
        ├── FPGA CV Accelerator: black-box image processing
        ├── Camera subsystem
        ├── Data logger
        └── Telemetry / UI streamer
```

### 4.2 Design Principle

The Raspberry Pi or embedded Linux computer is the main system coordinator.

The microcontroller handles real-time actuation and safety.

The FPGA is a replaceable accelerator module, not the robot brain.

---

## 5. Mechanical Subsystem

### 5.1 Mechanical Architecture

```text
Central chassis
├── 8 spring-loaded struts
├── 8 preload-reduction cables
├── central cable equalizer
├── main worm-geared winch or lead-screw actuator
├── 4 driven wheels
├── 4 passive support wheels
├── front camera mount
├── rear camera mount
├── front/rear LED rings
├── strut-mounted LED modules
└── tether strain relief / rear fairlead
```

### 5.2 Strut Function

Each strut should:

- maintain wheel contact with pipe wall,
- absorb pipe diameter variation,
- provide traction through spring normal force,
- allow preload reduction during turns,
- survive debris, vibration, and impact.

### 5.3 Central Cable-Preload System

The robot uses springs as the default outward force. Cables oppose the spring force.

```text
Spring pushes wheel outward → high normal force
Cable pulls strut inward → reduced normal force
```

Effective wheel normal force:

```text
F_normal ≈ F_spring - F_cable
```

### 5.4 Recommended Preload Mechanism

Preferred:

```text
Motor → worm gear or lead screw → sliding equalizer collar → 8 cables → struts
```

Avoid wrapping all 8 cables directly onto one spool unless an equalization mechanism is added.

### 5.5 Preload Modes

| Mode | Cable Tension | Effective Strut Force | Use |
|---|---:|---:|---|
| Inspection | Low | High | Stable imaging and traction |
| Transit | Medium | Medium | General driving |
| Turn | High | Low | Tank-turn / yaw rotation |
| Recovery | Pulsed | Variable | Free robot from minor jams |

### 5.6 Mechanical Fail-Safe

The safe default state should be:

```text
Cable released → spring force high → wheels contact pipe
```

If cable, actuator, or power fails, the robot should still have mechanical wall contact and be recoverable by tether.

---

## 6. Locomotion Strategy

### 6.1 Recommended V1 Locomotion

```text
4 driven rubber wheels
4 passive sprung support wheels
8 total struts
central preload reduction
differential tank steering
manual teleoperation
IMU-assisted yaw measurement
```

### 6.2 Tank Turn Sequence

For a right T-turn:

```text
1. Approach junction slowly.
2. Stop near branch opening.
3. Increase cable tension to reduce preload.
4. Command left wheels forward and right wheels reverse.
5. Monitor IMU yaw and camera feed.
6. Rotate toward branch.
7. Release preload to regain traction.
8. Drive into branch.
```

### 6.3 Avoided Locomotion Concepts

| Concept | Reason for Avoidance |
|---|---|
| 8 fixed wheels with no preload reduction | Difficult to tank-turn due to lateral scrub |
| Ball drive | High mechanical complexity, poor dirty-pipe robustness |
| Mecanum / omni wheels | Debris-sensitive, poor wet-pipe traction |
| Pure pneumatics | Cost, leakage, system complexity |
| Electromagnetic strut preload | Power, heat, nonlinear force, poor fail-safe behavior |

---

## 7. Lighting Subsystem

### 7.1 Lighting Problem

Central body lighting can be blocked by struts and wheels, producing shadows. These shadows can be misclassified as cracks by CV.

### 7.2 Recommended Lighting Architecture

```text
Front diffuse LED ring
Rear diffuse LED ring
4–8 fixed strut-mounted LED modules
Optional side illumination
```

Do not mount LEDs on rotating wheels. Mount them on fixed strut arms near the wheels.

### 7.3 Lighting Zones

| Zone | Purpose |
|---|---|
| Front ring | Forward navigation and inspection |
| Rear ring | Reverse driving and tether monitoring |
| Strut LEDs | Reduce shadows and improve wall illumination |
| Optional side LEDs | Pipe wall inspection |

### 7.4 Lighting Modes

| Mode | Front Ring | Rear Ring | Strut LEDs | Purpose |
|---|---:|---:|---:|---|
| Navigation | Medium | Low | Low | Driving |
| Inspection | High | Low | Medium/High | CV capture |
| Junction | High | Medium | High | Branch visibility |
| Reverse | Low | High | Medium | Recovery |
| Calibration | Fixed | Fixed | Fixed | Repeatable CV testing |

### 7.5 Lighting Design Rules

- Use diffusers.
- Avoid direct glare into cameras.
- Use constant-current LED drivers when possible.
- Use high PWM frequency if PWM is used.
- Test wet-surface glare early.
- Log lighting mode with each image frame.

---

## 8. Camera and Vision Subsystem

### 8.1 V1 Camera Configuration

Minimum:

```text
Front camera
Rear camera
```

Preferred:

```text
Front wide-angle camera
Rear camera
Optional side cameras
```

### 8.2 Camera Roles

| Camera | Role |
|---|---|
| Front | Navigation, crack detection, junction approach |
| Rear | Reverse driving, tether monitoring |
| Side | Wall inspection, optional |
| Fisheye | Junction awareness, optional |

### 8.3 Camera Pipeline

```text
Camera
→ Raspberry Pi capture
→ frame normalization
→ CV backend selection
    ├── OpenCV baseline
    └── FPGA accelerator
→ overlay mask/features
→ stream to UI
→ log raw and processed data
```

---

## 9. FPGA Black-Box Accelerator

### 9.1 FPGA Role

The FPGA performs accelerated image-processing tasks. It does not control the robot.

Allowed FPGA tasks:

- RGB to grayscale,
- crop / resize,
- blur / smoothing,
- Sobel edge detection,
- thresholding,
- morphology,
- binary crack-mask generation,
- feature pre-processing.

Not FPGA responsibilities:

- motor control,
- safety,
- teleoperation,
- UI,
- mission state,
- data logging,
- robot recovery.

### 9.2 FPGA API Philosophy

The rest of the robot sees the FPGA as:

```text
Input: image frame
Output: processed mask/features + metrics
```

The internals can be HDL, HLS, streaming logic, line buffers, or custom pipelines. The external API should remain stable.

### 9.3 FPGA Control API

Example control message:

```json
{
  "command": "fpga_set_mode",
  "mode": "sobel_threshold"
}
```

Example parameter message:

```json
{
  "command": "fpga_set_params",
  "params": {
    "threshold": 42,
    "blur_kernel": 3,
    "morphology": "close",
    "min_component_area": 20
  }
}
```

Example status response:

```json
{
  "fpga_ready": true,
  "mode": "sobel_threshold",
  "frame_width": 640,
  "frame_height": 480,
  "input_format": "grayscale8",
  "output_format": "binary_mask8",
  "last_frame_latency_ms": 4.8,
  "fps": 120,
  "temperature_C": 48.2,
  "power_W": 2.1,
  "fault": null
}
```

### 9.4 FPGA Data API

Recommended V1 input:

```text
Format: grayscale8
Resolution: 640 x 480
Stride: 640
Layout: row-major
```

Recommended V1 output:

```text
binary_mask8
0 = background
255 = crack candidate
```

Optional feature output:

```json
{
  "features": [
    {
      "x": 120,
      "y": 240,
      "area_px": 83,
      "length_px": 41,
      "orientation_deg": 12,
      "confidence": 0.72
    }
  ]
}
```

### 9.5 FPGA Transport Options

| Interface | Usefulness |
|---|---|
| USB | Good practical choice if board supports it |
| Ethernet | Best clean black-box boundary |
| SPI | Acceptable for control or low-resolution ROI |
| AXI/shared memory | Good for Zynq, less black-box-like |
| PCIe | High performance, likely too complex |

Recommended:

```text
Pi ↔ FPGA over USB or Ethernet
Control: JSON
Frame data: binary packets
Status: JSON
```

### 9.6 FPGA Fallback Requirement

If the FPGA fails:

- raw video must still stream,
- robot must still drive,
- Pi OpenCV baseline should still run if possible,
- operator must be able to recover the robot.

---

## 10. Embedded Control Subsystem

### 10.1 Recommended Controller

Use a real-time microcontroller:

- STM32,
- Teensy,
- RP2040,
- ESP32 if Wi-Fi is not relied upon for critical control.

### 10.2 Microcontroller Responsibilities

- motor PWM,
- encoder reading,
- motor current limits,
- preload actuator control,
- preload limit switches,
- lighting control,
- IMU reading,
- voltage/current telemetry,
- safety timeout,
- emergency stop.

### 10.3 Raspberry Pi Responsibilities

- camera capture,
- UI stream,
- high-level robot state machine,
- command routing,
- FPGA client,
- OpenCV baseline,
- data logging,
- telemetry aggregation.

### 10.4 Pi-to-MCU API Example

Drive command:

```json
{
  "cmd": "drive",
  "linear_mps": 0.05,
  "angular_radps": 0.0
}
```

Preload command:

```json
{
  "cmd": "set_preload",
  "mode": "turn"
}
```

Lighting command:

```json
{
  "cmd": "set_lighting_mode",
  "mode": "inspection"
}
```

MCU telemetry:

```json
{
  "status": "ok",
  "preload_percent": 65,
  "motor_current_A": [0.7, 0.8, 0.6, 0.7],
  "imu_yaw_deg": 34.2,
  "faults": []
}
```

For the first prototype, JSON over USB serial is acceptable. For the final version, add packet framing and CRC.

---

## 11. Electrical and Power Subsystem

### 11.1 Power Domains

```text
Main battery or tether power
        │
        ├── Motor rail: 12–24 V
        ├── LED rail: 12 V constant-current drivers
        ├── Compute rail: 5 V
        ├── FPGA board rail: board-specific
        └── Sensor/control rail: 3.3 V / 5 V
```

### 11.2 Protection Requirements

Include:

- fuse or resettable fuse,
- reverse-polarity protection,
- TVS diodes where appropriate,
- motor driver current limiting,
- bulk capacitance near motor drivers,
- separate logic and motor regulation,
- emergency motor cutoff,
- temperature monitoring,
- voltage and current telemetry.

### 11.3 Power Risk

Motors and LEDs can cause voltage sag. FPGA and Raspberry Pi are sensitive to brownout.

Mitigation:

- separate converters,
- star grounding,
- current sensing,
- bulk capacitors,
- power-on sequencing,
- stress test with motors + LEDs + compute active.

---

## 12. Communication and Tether

### 12.1 Recommended V1 Communication

Use Ethernet over tether.

Options:

```text
Ethernet + onboard battery
Ethernet + separate power wires
```

Fiber is a stretch goal unless the team has specific hardware experience.

### 12.2 Tether Requirements

- low-friction jacket,
- strain relief,
- rear fairlead,
- adequate bend radius,
- protected connector,
- visible in rear camera,
- does not interfere with struts.

### 12.3 Operator Link

Recommended:

```text
Video: MJPEG / RTSP / WebRTC
Commands: TCP / WebSocket
Telemetry: UDP / WebSocket
```

Prototype:

```text
MJPEG video stream + TCP JSON commands
```

---

## 13. Operator UI

### 13.1 Required UI Features

- front video,
- rear video,
- CV overlay,
- driving controls,
- preload mode display,
- lighting mode control,
- motor current display,
- yaw/pitch/roll,
- FPGA status,
- CV backend switch,
- record/log button,
- fault display,
- emergency stop.

### 13.2 UI Stack Options

| Option | Notes |
|---|---|
| Python + OpenCV window | fastest for prototype |
| Python + PyQt | good final desktop UI |
| Web dashboard | good cross-platform option |
| ROS 2 + RViz/rqt | powerful but may add overhead |

Recommended:

```text
3-month demo: Python/OpenCV + keyboard/controller input
8-month final: PyQt or web dashboard
```

---

## 14. Robot State Machine

The Raspberry Pi should run the high-level state machine.

```text
BOOT
→ SELF_TEST
→ IDLE
→ MANUAL_DRIVE
→ INSPECTION
→ TURN_PREP
→ TANK_TURN
→ TURN_RECOVERY
→ REVERSE_RECOVERY
→ FAULT
→ ESTOP
```

### 14.1 Inspection Mode

```text
1. Preload high.
2. Front and strut lighting active.
3. Camera capture active.
4. FPGA or OpenCV CV active.
5. Robot drives slowly.
6. Raw video, masks, and telemetry are logged.
```

### 14.2 Turn Mode

```text
1. Operator selects turn.
2. Lighting switches to junction mode.
3. Preload actuator tensions cables.
4. MCU confirms preload position.
5. Robot tank-turns at low speed.
6. IMU yaw is monitored.
7. Preload is released to regain traction.
8. Robot resumes driving.
```

---

## 15. Failure Modes and Mitigations

### 15.1 Mechanical Failure Modes

| Failure Mode | Effect | Mitigation |
|---|---|---|
| Robot stuck in pipe | Mission failure | rounded chassis, reverse mode, tether recovery |
| Wheel loses traction | cannot drive/turn | tune preload, high-friction wheels |
| Strut over-compresses | mechanical damage | hard stops, min pipe diameter spec |
| Cable snaps | one strut returns high preload | acceptable fail-safe |
| Cable jams | preload stuck | covered guides, limit switches, manual release |
| Winch jams pulled-in | low traction | spring return, access release |
| Uneven cable tension | robot tilts/pulls | equalizer plate, low-friction routing |
| Wheel jams on debris | motor stall | current sensing, wheel guards |
| Chassis too long | cannot T-turn | compact body, full-scale mockup |
| Tether snags | robot stuck | rear camera, fairlead, low-friction tether |

### 15.2 Lighting and Vision Failure Modes

| Failure Mode | Effect | Mitigation |
|---|---|---|
| Strut shadows | false cracks | strut LEDs, diffusers, multi-angle lighting |
| Glare from wet pipe | poor CV | angled LEDs, diffusers, exposure control |
| Mud on lens | poor visibility | lens cover, rear camera, cleaning protocol |
| LED flicker | bad frames | constant-current drivers or high PWM frequency |
| Wheel/strut in frame | occlusion | camera placement, FOV testing |
| CV false positive | incorrect defect report | test on shadows, stains, joints |
| CV false negative | missed crack | conservative thresholding, logging raw data |

### 15.3 Electrical and Control Failure Modes

| Failure Mode | Effect | Mitigation |
|---|---|---|
| Pi brownout | loss of UI/control | dedicated 5 V regulator |
| FPGA failure | CV unavailable | Pi baseline fallback |
| MCU timeout | unsafe control | watchdog and motor stop |
| Motor driver overheats | drive failure | current limit, thermal monitoring |
| LED overcurrent | lighting failure | constant-current drivers |
| Communication loss | robot unresponsive | timeout stop, tether recovery |
| Preload over-tension | loss of contact | current limit, hard stops |
| Power noise | resets/vision noise | separated rails, filtering |

---

## 16. Technical Stack Summary

| Subsystem | Recommended Stack |
|---|---|
| Mechanical | CAD, 3D printed / machined chassis, spring struts, Dyneema/steel cables |
| Preload | worm gear or lead screw, equalizer collar, limit switches |
| Drive | DC gear motors, encoders, H-bridge drivers |
| Motor control | STM32 / Teensy / RP2040 |
| Lighting | LED rings, strut LED boards, constant-current drivers |
| Cameras | USB or CSI cameras |
| Main compute | Raspberry Pi 5 / CM4 |
| Baseline CV | Python/C++ OpenCV |
| FPGA | HDL pipeline, black-box API |
| Robot software | Python or C++ supervisor |
| Communication | Ethernet tether, TCP/WebSocket/UDP |
| UI | Python/OpenCV first, PyQt/web final |
| Logging | video + telemetry + CV output + FPGA metrics |

---

# 17. 8-Month Roadmap

The roadmap assumes approximately 32 weeks total.

The project must have a demonstrable prototype by approximately Month 3, so the first 12 weeks are focused on a minimum viable robot rather than the full final system.

---

## Month 1: Requirements, Architecture, and Risk Reduction

### Goals

- Freeze V1 architecture.
- Define target pipe diameter.
- Select core electronics.
- Build early test rigs.
- Finalize FPGA black-box API contract.

### Tasks

#### Mechanical

- Define target pipe diameter range.
- Sketch chassis and strut geometry.
- Estimate strut travel and spring force.
- Design one-strut test rig.
- Design central preload concept.

#### Electrical

- Select motors and motor drivers.
- Select battery/tether power strategy.
- Select microcontroller.
- Select LED driver approach.
- Draft power budget.

#### Compute / CV

- Select Raspberry Pi or embedded Linux board.
- Set up camera capture.
- Implement OpenCV crack-detection baseline on sample images.
- Define CV processor interface.

#### FPGA

- Select FPGA board.
- Define FPGA API.
- Define first HDL target: grayscale + Sobel + threshold.
- Build software mock FPGA module.

#### Integration

- Define command/telemetry JSON.
- Define robot state machine.
- Set up Git repository and issue tracker.
- Define logging format.

### Month 1 Deliverables

- System architecture document.
- Subsystem interface document.
- One-strut force/preload test plan.
- FPGA API draft.
- Initial OpenCV baseline running on stored images.
- Bill of materials V1.

---

## Month 2: Subsystem Prototypes

### Goals

- Prove motor control.
- Prove one-strut preload mechanism.
- Prove lighting/camera visibility.
- Prove FPGA mock integration.

### Tasks

#### Mechanical

- Build one strut.
- Test spring force and travel.
- Build small cable-pull preload test.
- Measure force reduction versus cable displacement.
- Start full chassis CAD.

#### Electrical

- Run one motor with encoder.
- Run motor driver with current sensing.
- Run LED ring and strut LED sample.
- Assemble MCU breadboard or dev board wiring.

#### Compute / CV

- Camera live stream on Pi.
- Basic UI: display camera + keyboard/controller drive commands.
- Implement software CV backend interface.
- Integrate mock FPGA backend.

#### FPGA

- Simulate first image-processing HDL block.
- Create testbench using stored image data.
- Define frame input/output format.

#### Integration

- Pi-to-MCU serial communication.
- Basic telemetry packets.
- Basic command packets.

### Month 2 Deliverables

- One-strut preload demo.
- One-motor closed-loop control demo.
- Camera + lighting demo inside pipe-like mockup.
- Pi software can switch between OpenCV and mock FPGA.
- HDL block simulation result.

---

## Month 3: Minimum Viable Prototype Demo

### Goal

Produce a working prototype or demonstrator by the 3-month mark.

This does not need to be the final robot. It must prove the core integration path.

### Required 3-Month Demo Scope

The 3-month demo should show:

1. A partial or full chassis with multiple struts.
2. At least 4 wheels in contact with a pipe mockup.
3. Motorized forward/reverse movement.
4. Central preload mechanism reducing strut force.
5. Basic tank-turn or reduced-friction turn behavior.
6. Front camera live feed.
7. LED lighting.
8. Pi OpenCV baseline defect detection on test images or live feed.
9. FPGA API stub or early FPGA processing block.
10. Telemetry logging.

### Recommended 3-Month Demo Configuration

```text
Partial robot acceptable:
- 4 to 8 struts
- 2 to 4 driven wheels
- one central preload actuator
- front camera
- front LED ring
- optional strut LEDs
- Pi supervisor
- MCU motor/preload control
- OpenCV baseline
- FPGA mock or early FPGA Sobel block
```

### Month 3 Deliverables

- Physical mobility prototype.
- Preload softening demonstration.
- Camera and lighting demonstration.
- Baseline CV demo.
- FPGA black-box API demo with either mock or real FPGA.
- Logged telemetry from a test run.
- Updated risk register.

### Month 3 Success Criteria

Minimum acceptable:

```text
Robot moves inside a pipe mockup and shows live video.
Preload mechanism changes strut force.
Software stack can process frames through OpenCV and through FPGA API stub.
```

Strong demo:

```text
Robot performs a low-preload tank turn in a pipe mockup.
FPGA performs at least one real image-processing stage.
```

---

## Month 4: Full Mechanical and Electrical Integration

### Goals

- Move from prototype to full integrated V1 robot.
- Build all 8 struts.
- Integrate main preload actuator.
- Stabilize power and wiring.

### Tasks

#### Mechanical

- Fabricate full chassis.
- Install all 8 struts.
- Install central equalizer and 8 cables.
- Add tether strain relief.
- Add camera and LED mounts.
- Add protective covers.

#### Electrical

- Build robot wiring harness.
- Mount motor drivers.
- Mount LED drivers.
- Add fuses and power distribution.
- Add voltage/current sensing.
- Add limit switches.

#### Control

- Tune motor control.
- Tune preload actuator control.
- Add safety timeouts.
- Add fault handling.

#### Vision

- Improve lighting.
- Collect pipe mockup dataset.
- Tune OpenCV baseline.

#### FPGA

- Run real FPGA pipeline on stored frames.
- Start live-frame integration if hardware supports it.

### Month 4 Deliverables

- Full mechanical V1 robot assembled.
- Wiring harness V1.
- Safety stop working.
- Preload actuator homing and limits working.
- Lighting modes working.
- Improved CV dataset.

---

## Month 5: FPGA Hardware Integration and Benchmarking

### Goals

- Replace FPGA mock with real FPGA accelerator.
- Benchmark against Raspberry Pi baseline.
- Establish repeatable CV test procedure.

### Tasks

#### FPGA

- Implement grayscale / Sobel / threshold pipeline.
- Add morphology if feasible.
- Add performance counters.
- Add status telemetry.
- Validate output against OpenCV equivalent.

#### Software

- Integrate real FPGA client.
- Add backend switch: OpenCV vs FPGA.
- Log latency, FPS, CPU use, power.

#### Electrical

- Measure FPGA power.
- Measure Pi power under baseline CV.
- Add thermal monitoring.

#### Testing

- Run the same images/video through both CV paths.
- Compare masks and features.
- Compute FPS, latency, and energy/frame.

### Month 5 Deliverables

- Real FPGA backend integrated.
- FPGA vs Pi benchmark table.
- Power and latency measurements.
- CV output comparison images.

---

## Month 6: Junction Navigation and Robustness

### Goals

- Improve turning and recovery behavior.
- Build or refine T-junction test rig.
- Reduce self-shadowing and CV false positives.

### Tasks

#### Mechanical

- Test T-junction geometry.
- Adjust chassis corners and strut placement.
- Improve tether guide.
- Improve cable covers.

#### Control

- Implement turn-assist state sequence.
- Use IMU yaw feedback.
- Add motor-current-based stall detection.
- Add recovery wiggle mode.

#### Lighting / Vision

- Test strut shadows.
- Tune distributed lighting.
- Add CV calibration lighting mode.
- Test wet/glare conditions if safe.

#### FPGA / CV

- Tune thresholds and morphology.
- Add feature extraction if time allows.
- Improve overlay and logging.

### Month 6 Deliverables

- Tank-turn assist mode.
- T-junction test evidence.
- Recovery mode.
- Improved lighting layout.
- Reduced false-positive dataset results.

---

## Month 7: System Refinement and Final Demonstration Preparation

### Goals

- Stabilize system for final demo.
- Improve reliability.
- Complete documentation and test results.

### Tasks

- Run endurance tests.
- Run repeated pipe traversal tests.
- Run repeated preload cycles.
- Run motor stall safety tests.
- Run communication loss test.
- Finalize UI.
- Finalize data logging.
- Create inspection report output.
- Prepare final demo script.
- Prepare poster/report figures.

### Month 7 Deliverables

- Stable integrated robot.
- Final UI.
- Final benchmark results.
- Failure mode test results.
- Demo procedure.
- Draft final report.

---

## Month 8: Final Validation, Report, and Demo

### Goals

- Complete final testing.
- Freeze design.
- Deliver final capstone demonstration.

### Tasks

- Perform final pipe traversal demo.
- Perform final CV demo.
- Perform FPGA vs Pi comparison.
- Record final videos.
- Collect final telemetry.
- Complete final report.
- Complete presentation.
- Complete design review documentation.

### Month 8 Deliverables

- Final robot prototype.
- Final demo video.
- Final report.
- Final presentation.
- Final BOM.
- Final FMEA.
- FPGA API documentation.
- Test dataset and benchmark results.

---

# 18. 3-Month Prototype Scope Control

The 3-month prototype must avoid overreach.

## Must Have by Month 3

| Item | Required? |
|---|---|
| Pipe mockup | Yes |
| Moving robot or partial robot | Yes |
| At least 4 wheel contacts | Yes |
| Central preload mechanism demo | Yes |
| Basic remote control | Yes |
| Front camera live feed | Yes |
| Basic lighting | Yes |
| OpenCV baseline | Yes |
| FPGA API stub | Yes |
| Telemetry log | Yes |

## Nice to Have by Month 3

| Item | Priority |
|---|---|
| Full 8-strut chassis | Medium |
| Real FPGA Sobel block | High if feasible |
| Rear camera | Medium |
| Strut LEDs | Medium |
| T-junction turn | Medium |
| Full UI dashboard | Low |

## Do Not Require by Month 3

| Item | Reason |
|---|---|
| Fully waterproof robot | Too much scope |
| Full Y-junction navigation | Too hard early |
| Neural-network FPGA acceleration | Too risky |
| Fiber-optic tether | Integration overhead |
| Fully autonomous navigation | Not needed |
| Full defect classification | Detection mask is enough |

---

# 19. Suggested Team Allocation

## CE Student 1: FPGA Lead

Responsibilities:

- FPGA board setup,
- HDL pipeline,
- testbenches,
- FPGA API wrapper,
- performance counters,
- FPGA benchmark.

## CE Student 2: Software / CV Lead

Responsibilities:

- Raspberry Pi supervisor,
- OpenCV baseline,
- camera capture,
- FPGA client,
- UI,
- logging,
- benchmark scripts.

## EE Student 1: Power Lead

Responsibilities:

- battery/tether power,
- regulators,
- fuses,
- power distribution,
- voltage/current sensing,
- thermal monitoring.

## EE Student 2: Motor and Preload Control Lead

Responsibilities:

- motors,
- motor drivers,
- encoders,
- preload actuator,
- limit switches,
- MCU control.

## EE Student 3: Lighting and Sensor Lead

Responsibilities:

- LED rings,
- strut LEDs,
- LED drivers,
- IMU,
- camera power,
- sensor wiring.

## EE Student 4: Integration and Mechanical-Electrical Lead

Responsibilities:

- CAD support,
- strut assembly,
- cable routing,
- chassis integration,
- wiring harness,
- test rig.

---

# 20. Weekly Milestone Plan

## Weeks 1–4

| Week | Milestone |
|---:|---|
| 1 | Freeze V1 requirements and pipe diameter target |
| 2 | Select motors, Pi, MCU, FPGA board, cameras |
| 3 | One-strut CAD and FPGA API draft |
| 4 | OpenCV baseline on images, motor bench test started |

## Weeks 5–8

| Week | Milestone |
|---:|---|
| 5 | One-strut preload test rig built |
| 6 | Pi-to-MCU command link working |
| 7 | Camera + LED test in pipe mockup |
| 8 | FPGA HDL simulation of first block |

## Weeks 9–12

| Week | Milestone |
|---:|---|
| 9 | Partial chassis moves |
| 10 | Central preload demo installed |
| 11 | Camera + OpenCV live demo |
| 12 | 3-month demo: mobility + preload + vision + FPGA API |

## Weeks 13–16

| Week | Milestone |
|---:|---|
| 13 | Full chassis fabrication |
| 14 | Full 8-strut assembly |
| 15 | Full wiring harness |
| 16 | Integrated V1 robot drives |

## Weeks 17–20

| Week | Milestone |
|---:|---|
| 17 | Real FPGA data path integrated |
| 18 | FPGA vs Pi benchmark begins |
| 19 | FPGA status telemetry added |
| 20 | Benchmark report draft |

## Weeks 21–24

| Week | Milestone |
|---:|---|
| 21 | T-junction test rig |
| 22 | Turn-assist sequence |
| 23 | Lighting/shadow optimization |
| 24 | Recovery mode and stall handling |

## Weeks 25–28

| Week | Milestone |
|---:|---|
| 25 | Endurance testing |
| 26 | Final UI refinement |
| 27 | Final dataset and CV tuning |
| 28 | Draft final report and demo plan |

## Weeks 29–32

| Week | Milestone |
|---:|---|
| 29 | Final validation tests |
| 30 | Final demo recording |
| 31 | Final presentation and report |
| 32 | Capstone demo and handoff |

---

# 21. Critical Engineering Tests

## 21.1 Mechanical Tests

| Test | Purpose |
|---|---|
| Strut force vs displacement | validate spring selection |
| Cable tension vs preload reduction | validate preload system |
| Full 8-strut equalization | check uneven tension |
| Tank turn current test | check motor load reduction |
| Pipe diameter fit test | verify geometry |
| Tether pull test | verify strain relief |
| Endurance cycle test | check cable/spring fatigue |

## 21.2 Electrical Tests

| Test | Purpose |
|---|---|
| Motor stall current | size protection |
| LED current/thermal test | prevent overheating |
| Pi brownout test | validate regulator |
| FPGA power test | benchmark energy/frame |
| Full-load power test | motors + LEDs + compute |
| Emergency stop test | safety validation |
| Communication loss test | watchdog validation |

## 21.3 CV and FPGA Tests

| Test | Purpose |
|---|---|
| OpenCV static image test | baseline correctness |
| Live video CV test | real-time performance |
| Shadow false-positive test | lighting validation |
| Wet/glare test | robustness |
| FPGA image equivalence test | compare to OpenCV |
| FPGA latency test | benchmark |
| Energy/frame test | justify FPGA acceleration |

---

# 22. Final Demo Definition

A strong final demo should include:

1. Robot enters pipe mockup.
2. Operator sees front video.
3. Lighting mode changes from navigation to inspection.
4. Robot detects artificial crack or defect.
5. UI overlays CV result.
6. Operator switches between Pi CV and FPGA CV.
7. Telemetry shows FPS, latency, and power.
8. Robot approaches T-junction.
9. Preload reduces.
10. Robot performs tank-turn or partial branch alignment.
11. Preload increases again.
12. Robot reverses or exits safely.
13. Data log is exported.

---

# 23. Minimum Final Success Criteria

Even if some stretch goals fail, the project is successful if it demonstrates:

- reliable straight-pipe locomotion,
- adaptive preload reduction,
- camera and lighting integration,
- live operator control,
- Raspberry Pi CV baseline,
- real FPGA acceleration of at least one CV stage,
- measured performance/power comparison,
- documented failure modes and mitigations.

---

# 24. Highest-Risk Items

| Risk | Severity | Mitigation |
|---|---:|---|
| T-junction turning fails | High | demonstrate preload-assisted tank turn even if full junction entry is incomplete |
| FPGA integration late | High | use API stub early; keep Pi baseline working |
| Mechanical preload binds | High | one-strut and 8-strut test rigs early |
| Lighting causes CV false positives | Medium/High | distributed lighting tests by Month 3 |
| Power instability | High | separate rails and full-load power tests |
| Tether snagging | High | rear camera, fairlead, recovery procedure |
| Scope creep | High | enforce 3-month and 8-month milestones |

---

# 25. Recommended Immediate Next Actions

1. Choose target pipe diameter range.
2. Pick V1 robot diameter and chassis length.
3. Build one-strut preload test rig.
4. Select motor, spring, cable, and actuator.
5. Select Pi, MCU, FPGA board, and cameras.
6. Define the FPGA API in a short interface-control document.
7. Build a pipe mockup before finalizing chassis geometry.
8. Implement OpenCV baseline and mock FPGA processor immediately.
9. Assign one owner per subsystem.
10. Create the Month 3 demo checklist and do not expand it.

---

# 26. Project Positioning Statement

This capstone project is best positioned as:

> A remotely operated pipe inspection robot using adaptive spring-strut preload for junction maneuvering and FPGA-accelerated computer vision for real-time defect detection. The design emphasizes system integration, failure recovery, and measurable performance comparison between embedded software CV and hardware-accelerated CV.

---

# 27. Version Notes

This document describes the recommended V1 architecture:

```text
8 spring-loaded struts
single central cable-preload actuator
4 driven wheels
4 passive support wheels
front/rear cameras
front/rear/strut lighting
Raspberry Pi supervisor
microcontroller real-time control
FPGA black-box CV accelerator
Ethernet tether
manual teleoperation with assisted turn modes
```

Major design constraints:

- prioritize reliable mobility over full autonomy,
- keep FPGA separate but API-compatible,
- make all mechanisms fail recoverably,
- demonstrate real integration by Month 3,
- reserve Y-junction and advanced CV for stretch goals.
