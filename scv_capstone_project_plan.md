# SCV Capstone Project Plan

**Project name:** SCV — FPGA-Accelerated Selective Collection Vehicle with Relay-Based Teleoperation, Active Vision, and Smart Home Base  
**Team composition:** 2 Computer Engineers, 4 Electrical Engineers  
**Core requirement:** Host laptop control with joystick, home-base relay communication, FPGA-accelerated AI perception, autonomous/semi-autonomous collection, and docking/charging demonstration.

---

## 1. Project Summary

The SCV is a budget autonomous collection robot system. A host laptop provides the operator interface through **Gradio**, accepts joystick/gamepad input, displays video/telemetry, and sends commands to a stationary home base. The home base acts as a communication relay, docking target, and charging station. The mobile collector robot uses a camera turret, rangefinder, FPGA-accelerated AI perception on a PYNQ-Z2, and a scoop/collector mechanism to locate, classify, approach, and collect selected target objects.

The preferred demo target is:

> Collect quartz-like mineral samples while rejecting sticks, wood, and ordinary stones.

The project is designed to demonstrate embedded AI, FPGA acceleration, robotic teleoperation, relay-based communication, active perception, autonomous target approach, payload sensing, and docking/charging telemetry.

---

## 2. System-Level Architecture

```text
[Joystick/Gamepad]
        ↓ USB/Bluetooth
[Host Laptop: Gradio UI + Control Software]
        ↓ Wi-Fi / optional ELRS / serial radio
[Home Base Relay + Docking/Charging Station]
        ↓ Wi-Fi / optional ELRS / serial radio
[Collector Robot Main Controller]
        ├── [PYNQ-Z2 Vision/AI Processor]
        └── [Motor/Sensor/Servo MCU]
```

### Core design rule

The robot, home base, and laptop should be treated as separate networked agents with defined interfaces.

- The **host laptop** owns user control, dashboard, video display, and mission commands.
- The **home base** owns relay, docking marker/beacon, and charging telemetry.
- The **robot main controller** owns behavior, autonomy, target selection, and command arbitration.
- The **PYNQ-Z2** owns vision inference and FPGA acceleration.
- The **MCU** owns low-level motors, servos, rangefinder, load cell, and safety timeouts.

---

## 3. Reduced Subsystems

To reduce integration risk, the project is divided into **five major subsystems**.

| Subsystem | Main Role |
|---|---|
| 1. Host Laptop Control Station | Gradio UI, joystick input, telemetry, video, mission commands |
| 2. Home Base Relay + Dock | Communication relay, docking target, charging telemetry |
| 3. Robot Main Controller | State machine, autonomy, target selection, command routing |
| 4. Robot Perception Stack | PYNQ-Z2 camera processing, FPGA AI inference, target reports |
| 5. Robot Electromechanical Stack | Drive motors, turret, collector, sensors, power, charging contacts |

---

## 4. Subsystem 1: Host Laptop Control Station

### Purpose

The host laptop is the human-machine interface. It must support joystick-based control and provide a Gradio dashboard for video, telemetry, state display, and mission control.

### Required functions

- Read joystick/gamepad input.
- Send drive commands.
- Send mode commands.
- Send turret commands.
- Send collector commands.
- Send emergency stop.
- Display video stream.
- Display robot telemetry.
- Display AI target reports.
- Display charging/docking status.
- Log test data.

### Tech stack

| Function | Recommended Tech |
|---|---|
| UI | Gradio |
| Main language | Python |
| Joystick/gamepad input | `inputs`, `pygame`, `evdev`, or custom HID reader |
| Video display | OpenCV frame capture displayed in Gradio |
| Network messaging | UDP sockets or WebSocket |
| Telemetry parsing | JSON |
| Logs | CSV/JSON |
| Config | YAML/JSON |

### Note on Gradio and joystick control

Gradio is good for dashboards, buttons, sliders, telemetry, and video display. It is not naturally designed for low-latency joystick polling. Therefore, the host should use a **background joystick polling process/thread** that updates command values, while Gradio provides:

- mode buttons
- emergency stop button
- telemetry display
- video display
- manual override status
- command visualization

Recommended structure:

```text
Joystick polling thread
        ↓
Shared command state
        ↓
Network transmit loop
        ↓
Gradio UI displays command and telemetry state
```

### Key commands sent by laptop

```json
{
  "type": "drive_cmd",
  "payload": {
    "vx": 0.35,
    "wz": -0.10,
    "estop": false
  }
}
```

```json
{
  "type": "mode_cmd",
  "payload": {
    "mode": "AUTO_SEARCH"
  }
}
```

```json
{
  "type": "turret_cmd",
  "payload": {
    "pan_deg": 20,
    "tilt_deg": -15
  }
}
```

```json
{
  "type": "collector_cmd",
  "payload": {
    "action": "LOWER_AND_COLLECT"
  }
}
```

---

## 5. Subsystem 2: Home Base Relay + Dock

### Purpose

The home base is the stationary infrastructure node. It relays commands and telemetry, provides a docking target, and demonstrates charging/contact telemetry.

### Required functions

- Receive commands from laptop.
- Forward commands to robot.
- Receive telemetry from robot.
- Forward telemetry to laptop.
- Monitor charging voltage/current.
- Detect dock/contact state.
- Provide docking marker or beacon.
- Provide physical charging interface.
- Provide optional local emergency stop.

### Preferred tech stack

| Function | Recommended Tech |
|---|---|
| Controller | Raspberry Pi preferred; ESP32 acceptable |
| Network | Wi-Fi AP or Wi-Fi client bridge |
| Relay software | Python UDP relay |
| Dock marker | AprilTag / ArUco printed marker |
| Charging telemetry | INA219 or INA226 current/voltage sensor |
| Contact detection | GPIO switch, voltage detect, or current detect |
| Power input | Wall adapter |
| Charger | Commercial battery charger module |
| Optional beacon | LED or IR beacon |
| Safety | Physical stop button |

### Recommended choice

Use a **Raspberry Pi home base** if budget permits. It gives easier networking, logging, SSH debugging, and packet relay development.

### Home base responsibilities

The home base should remain simple. It should not perform object detection or path planning.

It should implement:

```text
Laptop packet received
        ↓
Validate packet
        ↓
Forward to robot
        ↓
Log or update link status
```

and:

```text
Robot telemetry received
        ↓
Forward to laptop
        ↓
Add base/dock telemetry if needed
```

### Dock status message

```json
{
  "src": "base",
  "dst": "laptop",
  "type": "dock_status",
  "seq": 2201,
  "time_ms": 184211,
  "payload": {
    "robot_docked": true,
    "charging": true,
    "charge_voltage_v": 12.4,
    "charge_current_a": 0.6
  }
}
```

---

## 6. Subsystem 3: Robot Main Controller

### Purpose

The robot main controller is the behavior coordinator. It receives commands from the home base, talks to the PYNQ-Z2 for AI target reports, talks to the MCU for low-level actuation/sensing, and runs the autonomy state machine.

### Required functions

- Receive manual control commands.
- Receive mission mode commands.
- Apply command priority and safety rules.
- Communicate with PYNQ-Z2.
- Communicate with motor/sensor MCU.
- Select targets.
- Run search, approach, collect, return, and docking states.
- Send telemetry to host laptop.
- Log robot-side data.

### Preferred tech stack

| Function | Recommended Tech |
|---|---|
| Controller | Raspberry Pi / Orange Pi / similar SBC |
| Main language | Python |
| State machine | Python enum/class-based FSM |
| Network to base | UDP JSON |
| Interface to PYNQ-Z2 | TCP/UDP JSON |
| Interface to MCU | USB serial |
| Logging | CSV/JSON |
| Optional marker detection | OpenCV |

### Robot behavior modes

```text
MANUAL
ASSISTED
AUTO_SEARCH
SCAN_TARGETS
APPROACH_TARGET
COLLECT
VERIFY_PAYLOAD
RETURN_TO_BASE
DOCK
CHARGE
FAULT
ESTOP
```

### Command priority

```text
1. Emergency stop
2. Manual override
3. Safety stop / link-loss stop
4. Docking safety
5. Autonomous mission behavior
```

### Recommended finite state machine

```text
BOOT
  ↓
MANUAL
  ↓
AUTO_SEARCH
  ↓
SCAN_TARGETS
  ↓
TARGET_SELECTED
  ↓
APPROACH_TARGET
  ↓
COLLECT
  ↓
VERIFY_PAYLOAD
  ↓
AUTO_SEARCH or RETURN_TO_BASE
  ↓
DOCK
  ↓
CHARGE
```

### Robot telemetry message

```json
{
  "src": "robot",
  "dst": "laptop",
  "type": "robot_status",
  "seq": 5402,
  "time_ms": 301220,
  "payload": {
    "mode": "AUTO_SEARCH",
    "battery_v": 11.7,
    "payload_g": 280,
    "target_class": "quartz",
    "target_confidence": 0.88,
    "target_range_m": 1.24,
    "estop": false,
    "fault": false
  }
}
```

---

## 7. Subsystem 4: Robot Perception Stack

### Purpose

The perception stack identifies candidate objects and classifies them using FPGA-accelerated inference on the PYNQ-Z2.

### Required functions

- Capture camera frames.
- Scan scene using camera turret.
- Detect candidate object regions.
- Crop candidate regions.
- Run small CNN/GEMM/convolution accelerator on FPGA.
- Classify object as target/non-target or quartz/stone/wood/background.
- Return target reports to robot main controller.
- Benchmark CPU-only inference against FPGA-accelerated inference.

### Preferred tech stack

| Function | Recommended Tech |
|---|---|
| FPGA board | PYNQ-Z2 |
| OS | PYNQ Linux |
| Main language | Python |
| Camera | USB camera |
| Frame processing | OpenCV + NumPy |
| FPGA interface | PYNQ overlay |
| Accelerator design | Vitis HLS / Vivado HLS |
| AI model | small quantized CNN or GEMM classifier |
| Communication | TCP/UDP JSON |
| Benchmarking | Python timing logs |

### Recommended AI scope

Avoid full YOLO as the baseline. Use:

```text
candidate crop → FPGA CNN classifier → target report
```

Recommended classes:

```text
target / non_target / background
```

or, for a stronger demo:

```text
quartz / ordinary_stone / wood / background
```

### Perception sequence

```text
1. Robot asks turret to scan angle.
2. Camera frame is captured.
3. Candidate blobs are detected using simple CV.
4. Candidate crops are classified by FPGA-accelerated CNN.
5. PYNQ returns class, confidence, and bounding box.
6. Robot main controller selects target.
7. Robot centers turret and activates rangefinder.
```

### Target report message

```json
{
  "src": "pynq",
  "dst": "robot",
  "type": "target_report",
  "seq": 901,
  "time_ms": 130442,
  "payload": {
    "class": "quartz",
    "confidence": 0.91,
    "bbox": [320, 180, 90, 75],
    "pan_deg": -20,
    "tilt_deg": -15,
    "valid": true
  }
}
```

### Required benchmark table

| Metric | CPU Baseline | FPGA Accelerated |
|---|---:|---:|
| Inference latency | TBD | TBD |
| FPS | TBD | TBD |
| Accuracy | TBD | TBD |
| End-to-end target selection latency | TBD | TBD |
| Power estimate | TBD | TBD |

---

## 8. Subsystem 5: Robot Electromechanical Stack

### Purpose

This subsystem physically drives, scans, collects, senses, powers, and docks the robot.

### Required functions

- Tracked drive.
- Motor control.
- Turret pan/tilt actuation.
- Range sensing.
- Collector actuation.
- Payload measurement.
- Battery monitoring.
- Docking contact interface.
- Low-level safety timeout.

### Preferred hardware stack

| Function | Recommended Hardware |
|---|---|
| Low-level controller | Arduino Mega / ESP32 / STM32 |
| Drive motors | 12 V brushed DC gear motors |
| Motor driver | BTS7960, Cytron MD10C/MDDS10, or equivalent |
| Chassis | Tracked chassis |
| Turret servos | Metal gear servos |
| Camera mount | Rigid 3D printed / laser-cut mount |
| Rangefinder | VL53L1X / VL53L0X ToF or UART laser rangefinder |
| Collector | Front scoop + servo gate |
| Payload sensor | Load cell + HX711 |
| Battery | Protected battery pack |
| Regulators | Buck converters for 5 V and 3.3 V |
| Docking contacts | Pogo pins / spring contacts |
| Docking alignment | Funnel, guide rails, magnet |
| Safety | Fuse, power switch, kill switch, watchdog timeout |

### Recommended collector

Use a **front scoop with servo gate** rather than a precision claw. It is more reliable for irregular stones/quartz blocks and less sensitive to alignment errors.

### MCU responsibilities

- Generate motor PWM.
- Generate servo PWM.
- Read rangefinder.
- Read load cell.
- Read battery voltage.
- Read dock/contact switch if installed.
- Enforce command timeout.
- Send low-level telemetry to robot main controller.

### MCU serial command examples

```json
{"cmd":"drive","left":0.25,"right":0.20}
```

```json
{"cmd":"turret","pan_deg":-25,"tilt_deg":-15}
```

```json
{"cmd":"collector","action":"collect"}
```

```json
{"cmd":"estop","active":true}
```

### MCU telemetry example

```json
{
  "type": "low_level_status",
  "range_m": 1.18,
  "payload_g": 310,
  "battery_v": 11.6,
  "dock_contact": false,
  "fault": false
}
```

---

## 9. Communication Architecture

### Preferred baseline

Use **Wi-Fi for the main control, telemetry, and video path**.

```text
Laptop ⇄ Home Base ⇄ Robot
```

The home base can act as the Wi-Fi access point. Both the laptop and robot connect to it.

### Optional enhancement

Use **ELRS or RC-style control** as a separate manual override or safety channel.

Recommended role of ELRS:

- emergency stop
- low-latency manual override
- backup manual drive

Avoid making ELRS the only communication layer because telemetry, video, and structured messages are easier over Wi-Fi.

### Recommended communication stack

| Link | Recommended Protocol |
|---|---|
| Laptop ⇄ Home Base | UDP JSON over Wi-Fi |
| Home Base ⇄ Robot | UDP JSON over Wi-Fi |
| Robot ⇄ PYNQ-Z2 | TCP/UDP JSON |
| Robot ⇄ MCU | USB serial newline-delimited JSON |
| Base ⇄ Charging sensor | I2C/GPIO |

---

## 10. Standard Message Envelope

All software agents should use a shared packet envelope.

```json
{
  "src": "laptop",
  "dst": "robot",
  "type": "mode_cmd",
  "seq": 1021,
  "time_ms": 123456,
  "payload": {
    "mode": "RETURN_TO_BASE"
  }
}
```

### Required envelope fields

| Field | Purpose |
|---|---|
| `src` | Sending agent |
| `dst` | Intended receiving agent |
| `type` | Message type |
| `seq` | Packet sequence number |
| `time_ms` | Sender timestamp |
| `payload` | Message-specific data |

### Main agents

```text
laptop
base
robot
pynq
mcu
```

### Main message types

| Type | Direction | Purpose |
|---|---|---|
| `heartbeat` | all | Link health |
| `drive_cmd` | laptop → robot | Manual velocity command |
| `mode_cmd` | laptop → robot | Change operating mode |
| `turret_cmd` | laptop/robot → mcu | Set pan/tilt or scan |
| `collector_cmd` | laptop/robot → mcu | Actuate collector |
| `estop_cmd` | laptop/base → robot | Emergency stop |
| `robot_status` | robot → laptop | Main robot telemetry |
| `target_report` | pynq → robot/laptop | AI target result |
| `range_report` | mcu → robot | Distance reading |
| `payload_status` | mcu/robot → laptop | Bin weight |
| `dock_status` | base/robot → laptop | Docking/charging status |
| `link_status` | base → laptop | Relay health |

---

## 11. Integration Strategy

### Integration principle

Build and verify each link in isolation before combining the full system.

### Phase 1: Communication chain

Goal:

```text
Joystick → Laptop Gradio host → Home Base → Robot → Motor command
```

Success criteria:

- Joystick input is visible in Gradio.
- Laptop sends packets to base.
- Base forwards packets to robot.
- Robot acknowledges packets.
- Robot sends telemetry back.
- E-stop command works.

### Phase 2: Low-level robot bring-up

Goal:

```text
Robot SBC → MCU → motors/servos/sensors
```

Success criteria:

- Motors drive from serial commands.
- Turret servos move to commanded angles.
- Collector actuates.
- Rangefinder returns data.
- Load cell returns data.
- Motor timeout stops robot if command stream is lost.

### Phase 3: PYNQ perception standalone

Goal:

```text
Camera → PYNQ-Z2 → FPGA classifier → target report
```

Success criteria:

- Camera frames are captured.
- Candidate regions are generated.
- FPGA classifier runs.
- Target reports are emitted.
- CPU-vs-FPGA benchmark exists.

### Phase 4: Turret + perception

Goal:

```text
Turret scan → camera frame → target classification → range measurement
```

Success criteria:

- Turret scans fixed angles.
- Candidate target is detected.
- Turret centers on target.
- Rangefinder measures target distance.
- Robot main controller receives target report.

### Phase 5: Approach and collect

Goal:

```text
Target report → approach target → collect → payload update
```

Success criteria:

- Robot turns toward selected target.
- Robot drives toward target using range estimate.
- Collector picks up object.
- Load cell detects payload increase.

### Phase 6: Return and dock

Goal:

```text
Return command → find home base → dock/contact → charging telemetry
```

Success criteria:

- Robot can return toward base marker/beacon.
- Robot aligns with docking funnel.
- Contact or charging state is detected.
- Laptop displays docked/charging status.

### Phase 7: Full demo

Goal:

```text
Manual control + autonomous collection + return-to-base + docking telemetry
```

Success criteria:

- Operator can switch between manual and autonomous modes.
- Robot scans for targets.
- Robot selects valid target.
- Robot collects target.
- Robot returns when commanded or when payload threshold is reached.
- Robot docks and reports charging/contact telemetry.

---

## 12. Time Frame

This assumes a typical two-term capstone schedule of approximately **24 weeks**. If the project is one term only, compress by removing stretch goals and treating autonomous docking as assisted docking.

### Weeks 1–2: Requirements and architecture freeze

Deliverables:

- Final project scope.
- Final subsystem ownership.
- Final communication architecture.
- Draft `scv_protocol.md`.
- Preliminary bill of materials.
- Risk register.

Exit criteria:

- Team agrees on five-subsystem architecture.
- Message envelope is defined.
- Baseline demo scenario is defined.

---

### Weeks 3–4: Procurement and bench prototypes

Deliverables:

- Order chassis, motors, motor drivers, servos, camera, sensors, battery, dock parts.
- Set up PYNQ-Z2.
- Set up Raspberry Pi/robot SBC.
- Set up Gradio development environment.
- Create first UDP message sender/receiver.
- Create first MCU serial motor test.

Exit criteria:

- Laptop can send a test packet.
- Robot SBC can receive a test packet.
- MCU can drive a motor from serial command.
- PYNQ-Z2 can run a simple Python/OpenCV test.

---

### Weeks 5–6: Teleoperation chain

Deliverables:

- Gradio dashboard v1.
- Joystick polling process.
- Laptop-to-base-to-robot packet relay.
- Robot-to-laptop telemetry.
- Emergency stop command.
- Basic motor control through full chain.

Exit criteria:

```text
Joystick → Laptop → Base → Robot → MCU → Motors
```

works reliably.

---

### Weeks 7–8: Mechanical platform and low-level sensing

Deliverables:

- Tracked chassis assembled.
- Motor driver mounted.
- Battery and regulators installed.
- Turret pan/tilt mount assembled.
- Collector prototype assembled.
- Rangefinder and load cell connected.
- MCU telemetry implemented.

Exit criteria:

- Robot drives manually.
- Turret moves.
- Collector moves.
- Rangefinder and load cell report values.
- Safety timeout works.

---

### Weeks 9–10: PYNQ vision and dataset

Deliverables:

- Camera connected to PYNQ.
- Initial dataset collected.
- Classes defined: target/non-target or quartz/stone/wood/background.
- Baseline CPU classifier running.
- Candidate crop extraction working.
- First PYNQ target report message implemented.

Exit criteria:

- PYNQ can classify saved images or live crops.
- PYNQ sends target reports to robot main controller.

---

### Weeks 11–12: FPGA acceleration

Deliverables:

- Small CNN/GEMM/convolution accelerator implemented.
- PYNQ overlay integrated.
- CPU-vs-FPGA benchmark started.
- Quantized or hardware-friendly model tested.
- Live inference path demonstrated.

Exit criteria:

- FPGA acceleration produces valid inference output.
- Benchmark table has preliminary data.

---

### Weeks 13–14: Turret scanning and target selection

Deliverables:

- Turret scan routine.
- Target centering routine.
- Rangefinder measurement after target confirmation.
- Target ranking based on confidence and range.
- Robot receives target selection.

Exit criteria:

- Robot can scan, find, center, classify, and range a target while stationary.

---

### Weeks 15–16: Approach and collection

Deliverables:

- Approach behavior using pan angle and range.
- Collector pickup sequence.
- Payload verification.
- Return trigger based on payload threshold.
- Manual override during autonomous approach.

Exit criteria:

- Robot can approach and collect at least one target object in a controlled arena.

---

### Weeks 17–18: Home base docking and charging telemetry

Deliverables:

- Docking funnel/guide rails built.
- Contact pads/pogo pins installed.
- Charging voltage/current sensor connected.
- Base dock telemetry integrated.
- Visual marker or beacon installed.

Exit criteria:

- Robot can physically dock with assistance.
- Laptop shows dock/contact/charging telemetry.

---

### Weeks 19–20: Semi-autonomous return-to-base

Deliverables:

- Return-to-base mode.
- Base marker/beacon detection.
- Slow docking approach.
- Dock failure timeout.
- Manual correction path.

Exit criteria:

- Robot can return toward base and perform assisted or semi-autonomous docking.

---

### Weeks 21–22: Full system integration

Deliverables:

- Integrated Gradio dashboard.
- Reliable mode switching.
- Clean log files.
- Demo arena finalized.
- Failure recovery procedures.
- Final FPGA benchmark results.
- System diagram finalized.

Exit criteria:

- Full demo sequence works at least several times under controlled conditions.

---

### Weeks 23–24: Final validation and presentation

Deliverables:

- Final report.
- Final poster/slides.
- Demo video.
- Performance tables.
- Risk analysis.
- Lessons learned.
- Resume-ready technical claims.

Exit criteria:

- Stable final demo.
- Documented CPU-vs-FPGA acceleration.
- Documented collection success rate.
- Documented link reliability.
- Documented docking/charging demonstration.

---

## 13. Baseline Demo Definition

The baseline demo should be achievable even if stretch goals fail.

### Baseline demo

```text
1. Operator opens Gradio dashboard.
2. Operator drives robot manually through laptop → base → robot.
3. Video and telemetry are visible.
4. Operator starts autonomous scan.
5. Turret scans for target objects.
6. PYNQ classifies object of interest.
7. Robot centers target and measures range.
8. Robot approaches target.
9. Robot collects target with scoop/gate.
10. Load cell detects payload.
11. Operator or payload threshold triggers return-to-base.
12. Robot approaches docking station.
13. Robot physically docks or assisted-docks.
14. Laptop displays docked/charging telemetry.
```

---

## 14. Stretch Goals

Only attempt these after the baseline works.

| Stretch Goal | Risk | Value |
|---|---:|---:|
| Fully autonomous multi-object collection | Medium | High |
| Autonomous docking without manual correction | High | High |
| ELRS manual override channel | Medium | Medium-high |
| Multiple robots using same base protocol | High | High |
| Live object bounding boxes in Gradio | Medium | Medium |
| Automatic resume after charging | Medium | Medium |
| Full object detector on FPGA | High | High |
| Map/log of collected target positions | Medium | Medium |

---

## 15. Major Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---:|---|
| Too many subsystem boundaries | High | Use five-subsystem architecture |
| Gradio not suitable for joystick polling | Medium | Use background joystick thread/process |
| PYNQ integration takes too long | High | Keep CPU classifier as fallback |
| FPGA model too complex | High | Use small CNN/GEMM classifier, not YOLO baseline |
| Docking alignment unreliable | High | Use funnel/rails, magnet, assisted docking fallback |
| Rangefinder misses target | Medium | Use medium-sized targets and align rangefinder with camera |
| Wireless link instability | High | Start with Wi-Fi AP at home base; add ELRS only as optional |
| Collector misses irregular rocks | Medium | Use scoop/gate instead of precision claw |
| Battery/charging safety issue | High | Use commercial charger/BMS and current-limited charging |
| Integration happens too late | High | Teleoperation chain must work by weeks 5–6 |

---

## 16. Team Ownership

| Member | Suggested Role |
|---|---|
| Computer Engineer 1 | PYNQ-Z2, FPGA accelerator, AI model, inference benchmark |
| Computer Engineer 2 | Gradio host UI, protocol, relay software, robot state machine |
| Electrical Engineer 1 | Drive motors, motor drivers, battery distribution |
| Electrical Engineer 2 | Charging dock, contact system, power telemetry |
| Electrical Engineer 3 | Turret, rangefinder, load cell, sensor calibration |
| Electrical Engineer 4 | Collector mechanism, chassis integration, docking funnel, wiring harness |

---

## 17. Final Recommended Tech Stack

```text
Host Laptop:
Python + Gradio + joystick polling thread + OpenCV + UDP JSON + CSV/JSON logs

Home Base:
Raspberry Pi + Wi-Fi AP + Python UDP relay + INA219/INA226 + AprilTag/ArUco marker + charging dock

Robot Main:
Raspberry Pi/Orange Pi + Python FSM + UDP JSON + TCP/UDP to PYNQ + USB serial to MCU

PYNQ-Z2:
PYNQ Linux + OpenCV + NumPy + small CNN/GEMM accelerator + PYNQ overlay + JSON target reports

Low-Level Robot:
Arduino Mega/ESP32/STM32 + motor PWM + servo PWM + rangefinder + load cell + battery telemetry + watchdog safety

Mechanical:
Tracked chassis + front scoop/servo gate + pan/tilt camera turret + docking funnel + pogo/spring contacts + magnet

Communication:
Wi-Fi baseline through home base; optional ELRS/RC channel for manual override and emergency stop
```

---

## 18. Final Project Title Options

1. **SCV: FPGA-Accelerated Selective Collection Vehicle with Smart Home Base**
2. **Relay-Operated FPGA-AI Collection Robot with Active Vision and Autonomous Docking**
3. **Budget FPGA-AI Selective Material Collection Robot with Gradio-Based Teleoperation**
4. **Smart Collection Vehicle with FPGA Vision, Relay Control, and Docking Base**

Recommended title:

> **SCV: FPGA-Accelerated Selective Collection Vehicle with Relay-Based Teleoperation and Smart Docking**

---

## 19. Success Criteria

The final project should be considered successful if it demonstrates:

- Gradio host dashboard with joystick/manual control.
- Laptop-to-home-base-to-robot command path.
- Robot-to-home-base-to-laptop telemetry path.
- Live video or image feedback.
- FPGA-accelerated AI classification on PYNQ-Z2.
- Target/non-target classification.
- Active turret scan and target centering.
- Rangefinder-assisted approach.
- Physical collection of target object.
- Payload sensing.
- Return-to-base behavior or command.
- Dock/contact/charging telemetry.
- Emergency stop and link-loss safety behavior.
- CPU-vs-FPGA benchmark results.

---

## 20. Scope Control Rules

To keep the project achievable:

- Do not make full SLAM a baseline requirement.
- Do not make full YOLO-on-FPGA a baseline requirement.
- Do not require perfect autonomous docking for project success.
- Do not make ELRS the only communication path.
- Do not rely on a precision claw as the only collector.
- Do not let the home base perform autonomy.
- Do not let the laptop send raw motor PWM.
- Do not let the PYNQ directly control motors.
- Freeze the message protocol early.
- Integrate communication and manual control before AI and docking.

