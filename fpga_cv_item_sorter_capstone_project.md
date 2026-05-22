# FPGA-Accelerated Computer Vision Item Sorting System

## 1. Project Title

**FPGA-Accelerated Real-Time Conveyor Item Sorting System with Host-Laptop Visualization**

---

## 2. Project Summary

This capstone project aims to design and implement an intelligent conveyor-based sorting system. Randomly dumped items are transported on a conveyor belt, detected using a camera-based vision system, classified into categories, tracked in real time, and physically separated into output bins using an actuator.

The core technical contribution is an **FPGA-accelerated computer vision pipeline** implemented in HDL. The FPGA performs real-time image preprocessing, segmentation, object localization, and potentially classification support. A host laptop connects to the embedded system through USB or Ethernet to provide live video visualization, detection overlays, logging, calibration, and dataset collection.

The final system should demonstrate:

- Real-time item detection on a moving conveyor
- FPGA/HDL-based image processing acceleration
- Host-laptop live video monitoring and debugging
- Encoder-based item tracking
- Actuator timing synchronized to conveyor motion
- Physical sorting into multiple categories
- Quantitative validation of accuracy, latency, throughput, and FPGA resource usage

---

## 3. High-Level System Description

The system receives a random set of small objects through a hopper, chute, or manual dump area. Items are carried along a conveyor belt through a camera field of view. A camera captures frames of the belt and sends pixel data to the FPGA or SoC FPGA.

The FPGA processes the video stream to identify objects and extract localization information such as bounding boxes, centroids, and object areas. A classifier determines the item category. A tracking and scheduling module estimates when each object will reach the actuator region. The actuator then redirects, picks, or removes the object into the correct bin.

A host laptop displays the camera feed, processed video, detection overlays, telemetry, logs, and system controls.

---

## 4. Main Project Objectives

### 4.1 Primary Objectives

1. Design a conveyor-based sorting platform for randomly placed small objects.
2. Implement an FPGA-accelerated vision pipeline for real-time image processing.
3. Detect, localize, classify, and track multiple items on a moving belt.
4. Synchronize sorting actuation with conveyor encoder feedback.
5. Provide a host-laptop interface for live video, overlays, logging, and calibration.
6. Quantitatively evaluate system performance.

### 4.2 Stretch Objectives

1. Implement a lightweight fixed-point classifier or CNN accelerator on FPGA.
2. Add magnetic pickup for ferromagnetic items.
3. Implement multi-object scheduling and conflict handling.
4. Add automatic calibration routines.
5. Support multiple actuator modules.
6. Benchmark FPGA acceleration against software-only OpenCV processing.

---

## 5. System Architecture

```text
Random Item Input
      ↓
Feeder / Chute / Hopper
      ↓
Conveyor Belt with Encoder
      ↓
Camera + Controlled Lighting
      ↓
FPGA / SoC FPGA Vision Pipeline
      ↓
Object Detection + Classification + Tracking
      ↓
Actuation Scheduler
      ↓
Servo / Solenoid / Magnet / Diverter
      ↓
Sorted Output Bins
```

Host-laptop connection:

```text
FPGA / SoC FPGA / MCU
      ↕ USB or Ethernet
Host Laptop Dashboard
      - Live video display
      - Bounding-box overlays
      - Detection logs
      - Calibration interface
      - Dataset capture
      - System controls
```

---

## 6. Subsystem Breakdown

## 6.1 Material Handling Subsystem

### Purpose

Accept randomly dumped items and deliver them to the conveyor in a manageable, partially separated stream.

### Responsibilities

- Receive multiple randomly placed objects.
- Reduce item overlap.
- Prevent jams.
- Feed items consistently onto the belt.
- Support repeatable testing.

### Possible Designs

#### Option A: Manual Dump Zone

A simple input region where users place or dump items onto the belt.

**Advantages**
- Easy to build
- Good for early testing
- Low mechanical risk

**Disadvantages**
- Less realistic
- Randomness depends on user behavior

#### Option B: Passive Chute

A sloped chute feeds items onto the conveyor.

**Advantages**
- Simple
- Low-cost
- Better repeatability than manual dumping

**Disadvantages**
- Can still cause item clustering
- May need rails or spacing features

#### Option C: Hopper with Gate

A hopper holds items and releases them through an adjustable gate.

**Advantages**
- More realistic industrial design
- Better random-input demonstration
- Stronger mechanical subsystem

**Disadvantages**
- Higher chance of jamming
- More difficult to tune

### Recommended Design

Use a **small passive chute or hopper with an adjustable gate**. Add side rails and optional passive spreading features to reduce item overlap.

---

## 6.2 Conveyor Subsystem

### Purpose

Move items through the imaging zone and actuation zone at a controlled, measurable speed.

### Main Components

- Belt
- Rollers
- Motor
- Motor driver
- Encoder
- Frame
- Belt tensioner
- Side rails
- Output bins

### Requirements

- Smooth belt motion
- Adjustable speed
- Encoder feedback
- High-contrast belt surface
- Rigid camera and actuator mounting
- Minimal slip between motor and belt

### Recommended Implementation

- DC gearmotor with quadrature encoder, or stepper motor with known motion
- Closed-loop speed control if using DC motor
- Matte black or white belt depending on item contrast
- Encoder-based position tracking for item scheduling

---

## 6.3 Vision and Illumination Subsystem

### Purpose

Capture consistent images of items on the conveyor for detection and classification.

### Main Components

- Top-down camera
- Camera mount
- LED lighting
- Diffuser
- Shielding from ambient light
- Calibration target

### Requirements

- Fixed camera position
- Minimal motion blur
- Stable exposure
- Uniform lighting
- Known camera-to-belt geometry

### Camera Options

#### Option A: FPGA-Compatible Camera

Examples:
- Parallel CMOS camera
- MIPI CSI-2 camera if board supports it
- HDMI camera module with capture interface

**Advantages**
- Strong FPGA integration
- Real-time streaming pipeline
- Better final architecture

**Disadvantages**
- More difficult hardware integration

#### Option B: USB Webcam to Laptop

**Advantages**
- Easy video display
- Easy OpenCV prototyping
- Easy dataset collection

**Disadvantages**
- Weakens FPGA role if used as final perception path
- Requires frame transfer to FPGA if FPGA must process same images

#### Option C: SoC FPGA Camera Input

Best option if using a Zynq or similar SoC FPGA platform with supported camera interface.

### Recommended Approach

Use a two-stage development path:

1. **Prototype:** USB webcam connected to laptop for OpenCV algorithm development and dataset capture.
2. **Final system:** Camera connected to FPGA or SoC FPGA, with processed video streamed to host laptop over Ethernet or USB.

---

## 6.4 FPGA Vision Subsystem

### Purpose

Provide deterministic, low-latency image processing using custom HDL modules.

### Main FPGA Responsibilities

- Camera interface
- Pixel stream handling
- Frame or line buffering
- Color-space conversion
- Thresholding
- Morphological filtering
- Connected-component labeling
- Bounding-box extraction
- Centroid extraction
- Feature extraction
- Optional classifier acceleration
- Optional actuation timing support

### Proposed HDL Pipeline

```text
Camera Pixel Stream
      ↓
Input Synchronization
      ↓
RGB to Grayscale / HSV Conversion
      ↓
Thresholding / Segmentation
      ↓
Erosion / Dilation / Morphological Filtering
      ↓
Connected-Component Labeling
      ↓
Bounding Box + Centroid Extraction
      ↓
Feature Extraction
      ↓
AXI / FIFO / Register Interface
      ↓
Embedded Processor or MCU
```

### HDL Novelty Opportunities

- Streaming connected-component labeling
- Real-time bounding-box extraction
- Fixed-point color/shape feature extraction
- Low-latency FPGA object tracker
- FPGA-based actuation scheduler
- Lightweight quantized classifier
- FPGA-vs-CPU latency comparison

### Recommended FPGA Scope

The minimum strong FPGA deliverable should include:

- Streaming thresholding
- Morphological filtering
- Connected-component labeling
- Bounding-box and centroid extraction
- Host-readable object metadata

A strong stretch goal is:

- Fixed-point classifier or compact CNN accelerator

---

## 6.5 Object Detection and Classification Subsystem

### Purpose

Determine what each object is and where it is located.

### Inputs

- FPGA-generated bounding boxes
- Object masks
- Centroids
- Area
- Shape features
- Color features
- Optional image crops

### Outputs

- Object ID
- Category
- Confidence score
- Bounding box
- Centroid
- Scheduled actuation position

### Classification Methods

#### Classical Feature-Based Classification

Uses:
- Color histogram
- Area
- Aspect ratio
- Circularity
- Shape moments
- Edge density
- Texture features

**Advantages**
- FPGA-friendly
- Easier to debug
- Good for controlled item categories

**Disadvantages**
- Less robust for arbitrary objects

#### Lightweight CNN Classification

Uses a small neural network trained offline.

**Advantages**
- Better visual classification
- More modern CV component

**Disadvantages**
- More difficult to implement on FPGA
- Requires dataset and training pipeline

### Recommended Initial Item Categories

Use physically and visually distinct items:

- Red plastic part
- Blue plastic part
- Steel washer
- Brass or aluminum part
- 3D-printed shape
- Reject/unknown item

For a magnetic extension:

- Ferromagnetic metal
- Non-ferromagnetic metal
- Plastic
- Paper/cardboard

---

## 6.6 Tracking and Scheduling Subsystem

### Purpose

Track detected items as they move from the camera zone to the actuator zone.

### Responsibilities

- Assign object IDs
- Use conveyor encoder feedback
- Predict arrival at actuator
- Schedule actuation commands
- Reject uncertain detections
- Resolve object conflicts

### Encoder-Based Tracking

Each object should be stored with:

```text
object_id
category
confidence
centroid_pixel
centroid_world_mm
bbox
detection_encoder_tick
scheduled_actuation_tick
actuation_status
```

Actuation should be based on encoder displacement, not only elapsed time.

```text
scheduled_actuation_tick =
    detection_encoder_tick
    + camera_to_actuator_distance_ticks
    - actuator_delay_ticks
```

### Conflict Handling

If two items are too close together, the system may:

- Sort the higher-confidence item
- Reject both items
- Slow the conveyor
- Pause the conveyor
- Mark the region as invalid
- Trigger a reject-bin mechanism

---

## 6.7 Actuation Subsystem

### Purpose

Physically separate objects into category-specific bins.

### Candidate Actuator Types

#### Servo Diverter Gate

A servo moves a gate to redirect items.

**Advantages**
- Simple
- Reliable
- Good for capstone feasibility
- Easy to synchronize

**Disadvantages**
- Limited speed
- Works best with separated items

#### Solenoid Diverter

A solenoid rapidly pushes or deflects an item.

**Advantages**
- Fast
- Simple control
- Good for high-speed sorting

**Disadvantages**
- Requires careful driver design
- Can be mechanically harsh

#### Electromagnet Pickup

An electromagnet picks up ferromagnetic objects.

**Advantages**
- Interesting technical extension
- Useful for metal sorting
- Strong demo value

**Disadvantages**
- Only works on ferromagnetic items
- Requires current control and flyback protection
- Release timing may be nontrivial

#### Robotic Arm

A robotic arm picks items from a moving conveyor.

**Advantages**
- Visually impressive
- Strong robotics component

**Disadvantages**
- High risk
- Moving-belt picking is difficult
- May distract from FPGA/CV focus

### Recommended Actuator

Use a **servo or solenoid diverter gate** as the baseline actuator.

Add an **electromagnet pickup** only as a stretch feature.

---

## 6.8 Embedded Control Subsystem

### Purpose

Coordinate the FPGA, motor, encoder, actuator, and host-laptop dashboard.

### Architecture Options

#### Option A: SoC FPGA

```text
FPGA Fabric:
  Vision pipeline

ARM Processor:
  Scheduling
  Communication
  Logging
  Optional actuator control
```

#### Option B: FPGA + Microcontroller

```text
FPGA:
  Vision pipeline

MCU:
  Motor control
  Encoder reading
  Actuator commands
  USB serial communication
```

#### Option C: FPGA + Host Laptop + MCU

```text
FPGA:
  Vision acceleration

Laptop:
  Dashboard and dataset collection

MCU:
  Motor and actuator control
```

### Recommended Architecture

Use a **SoC FPGA** if available. Otherwise use:

```text
FPGA + STM32/RP2040 MCU + Host Laptop
```

---

## 6.9 Host Laptop Dashboard Subsystem

### Purpose

Provide real-time visibility, debugging, calibration, and data logging.

### Responsibilities

- Display live camera feed
- Show processed mask
- Overlay bounding boxes and labels
- Display object tracking state
- Display actuator queue
- Log detection and sorting results
- Save datasets
- Send configuration commands
- Run calibration routines

### Recommended Software Stack

```text
Python
OpenCV
NumPy
PySerial or socket
PyQt / PySide / Dear PyGui
CSV or SQLite logging
PyTorch for offline model training
```

### Dashboard Views

- Raw camera feed
- Processed segmentation mask
- Bounding-box overlay
- Object table
- Conveyor telemetry
- Actuator queue
- Sorting statistics
- Calibration controls
- Dataset recording panel

---

## 6.10 Communication Subsystem

### Purpose

Transfer video, metadata, commands, and logs between embedded hardware and the host laptop.

### Communication Options

#### USB Serial

Good for:
- Metadata
- Control commands
- Telemetry
- Low-rate logs

Not ideal for:
- Full-resolution video

#### USB Bulk Transfer

Good for:
- Higher-rate data
- Frame transfer
- Debug video

More difficult than USB serial.

#### Ethernet

Good for:
- Video streaming
- Metadata
- Host dashboard
- Embedded Linux systems

Recommended for SoC FPGA platforms.

### Recommended Communication Stack

If using SoC FPGA with Linux:

```text
Ethernet:
  video stream + metadata + commands
```

If using FPGA + MCU:

```text
USB serial:
  metadata + control

USB webcam or Ethernet:
  video feed
```

### Example Metadata Packet

```json
{
  "type": "detection",
  "frame_id": 4218,
  "encoder_tick": 88422,
  "objects": [
    {
      "id": 12,
      "class_id": 2,
      "class_name": "washer",
      "confidence": 0.87,
      "bbox": [144, 92, 37, 31],
      "centroid": [162, 107],
      "scheduled_tick": 90110
    }
  ]
}
```

### Example Command Packet

```json
{
  "type": "config",
  "belt_speed_mm_s": 220,
  "threshold_low": 48,
  "threshold_high": 180,
  "mode": "run"
}
```

---

## 6.11 Power Electronics Subsystem

### Purpose

Provide safe and stable power to the FPGA, motors, actuators, sensors, camera, and lighting.

### Power Domains

| Domain | Typical Voltage | Loads |
|---|---:|---|
| FPGA board | Board-specific | SoC FPGA |
| Logic | 3.3 V / 5 V | Sensors, MCU |
| Motor | 12 V / 24 V | Conveyor motor |
| Actuator | 5 V / 12 V / 24 V | Servo, solenoid, magnet |
| Lighting | 5 V / 12 V | LED strip or panel |

### Required Protection

- Flyback diodes for solenoids and magnets
- MOSFET drivers for high-current loads
- Common ground strategy
- Fuses or resettable polyfuses
- Emergency stop
- Current sensing for magnet or solenoid
- Voltage regulation
- Electrical isolation if required

---

## 6.12 Mechanical Frame and Safety Subsystem

### Purpose

Provide a stable, safe, and repeatable platform.

### Requirements

- Rigid camera mount
- Rigid actuator mount
- Conveyor frame
- Adjustable lighting mount
- Fixed bin positions
- Belt guards
- Emergency stop
- Cable routing
- Mechanical limit stops
- Safe access to test area

### Safety Features

- Emergency stop button
- Software stop command
- Motor driver enable control
- Actuator timeout
- Current limit
- Mechanical guards near rollers
- Shielding around fast-moving actuators

---

## 6.13 Calibration Subsystem

### Purpose

Map camera measurements to physical conveyor coordinates and ensure correct actuation timing.

### Required Calibrations

#### Camera Calibration

- Pixel-to-mm mapping
- Camera alignment
- Lens distortion correction if necessary

#### Conveyor Calibration

- Encoder counts per millimeter
- Speed measurement
- Belt slip characterization

#### Actuator Calibration

- Command-to-motion delay
- Effective contact position
- Gate travel time
- Release delay for magnet

#### Camera-to-Actuator Calibration

Measure physical distance between:
- Camera detection line
- Actuator effective sorting line

---

## 7. Recommended Technology Stack

## 7.1 Final Recommended Stack

### FPGA / Embedded

| Layer | Recommended Technology |
|---|---|
| FPGA board | Zynq-7000 SoC FPGA, PYNQ-Z2, Zybo Z7, or equivalent |
| HDL | SystemVerilog or Verilog |
| FPGA tools | Vivado / Vitis, or vendor equivalent |
| Camera input | FPGA-compatible camera or SoC-supported camera |
| Processor | ARM processor on SoC FPGA |
| Embedded software | C/C++ or Python on embedded Linux |
| FPGA interface | AXI Stream, AXI Lite, DMA, FIFO, registers |

### Host Laptop

| Layer | Recommended Technology |
|---|---|
| Language | Python |
| Video processing | OpenCV |
| GUI | PyQt, PySide, or Dear PyGui |
| Communication | Ethernet sockets, PySerial, or USB |
| Logging | CSV or SQLite |
| ML training | PyTorch |
| Dataset tools | OpenCV, NumPy, LabelImg/CVAT optional |

### Motor and Actuation

| Function | Recommended Technology |
|---|---|
| Conveyor motor | DC gearmotor with encoder or stepper motor |
| Motor driver | H-bridge or stepper driver |
| Encoder | Quadrature encoder |
| Actuator | Servo or solenoid diverter |
| Magnet stretch | Electromagnet + MOSFET driver |
| Low-level controller | SoC FPGA ARM core, STM32, or RP2040 |

### Communication

| Data Type | Recommended Link |
|---|---|
| Metadata | USB serial or TCP socket |
| Commands | USB serial or TCP socket |
| Video stream | Ethernet, USB bulk, or SoC video stream |
| Debug logs | Serial, TCP, CSV on laptop |

---

## 8. Development Repository Structure

```text
fpga-cv-sorter/
  README.md

  fpga/
    rtl/
      camera_if.sv
      rgb_to_hsv.sv
      threshold.sv
      morphology.sv
      connected_components.sv
      bbox_extractor.sv
      feature_extractor.sv
      axi_stream_if.sv
      scheduler.sv

    sim/
      testbenches/
      sample_frames/
      expected_outputs/

    constraints/
      pins.xdc

    vivado/
      project_files/

  embedded/
    src/
      main.c
      motor_control.c
      encoder.c
      actuator.c
      scheduler.c
      comms.c
      safety.c

    include/
      motor_control.h
      encoder.h
      actuator.h
      scheduler.h
      comms.h
      safety.h

  host/
    dashboard/
      main.py
      video_view.py
      serial_client.py
      socket_client.py
      overlay.py
      logger.py
      calibration.py

    training/
      collect_dataset.py
      train_classifier.py
      export_model.py
      evaluate_model.py

    analysis/
      latency_benchmark.py
      sorting_accuracy.py
      fpga_vs_cpu.py

  mechanical/
    cad/
    drawings/
    stl/
    bom.csv

  electrical/
    schematics/
    wiring_diagrams/
    power_budget.xlsx

  docs/
    architecture.md
    interface_protocol.md
    calibration.md
    test_plan.md
    safety.md
    final_report.md
```

---

## 9. Key Interfaces

## 9.1 FPGA to Embedded Processor

### Interface Options

- AXI Lite registers
- AXI Stream
- DMA frame buffers
- FIFO
- Interrupt lines

### Data Exchanged

- Frame ID
- Object count
- Bounding boxes
- Centroids
- Features
- Class ID
- Confidence
- Encoder count
- Status flags

---

## 9.2 Embedded Processor to Host Laptop

### Interface Options

- USB serial
- TCP socket
- UDP socket
- WebSocket
- Ethernet video stream

### Data Exchanged

- Live video frames
- Processed masks
- Detection metadata
- System telemetry
- Sorting logs
- Configuration commands

---

## 9.3 Embedded Controller to Actuator

### Interface Options

- GPIO
- PWM
- MOSFET gate drive
- Servo PWM
- SPI/UART to motor controller

### Data Exchanged

- Actuation trigger
- Gate position
- Magnet enable
- Motor speed command
- Safety enable
- Fault status

---

## 10. Performance Metrics

The final system should be evaluated using measurable metrics.

| Metric | Target / Measurement |
|---|---|
| Detection accuracy | Percentage of objects correctly detected |
| Classification accuracy | Percentage of detected objects correctly classified |
| Sorting accuracy | Percentage of objects placed in correct bin |
| Throughput | Items sorted per minute |
| End-to-end latency | Camera capture to actuator command |
| FPGA latency | Pixel input to bounding-box output |
| FPGA resource usage | LUTs, FFs, BRAM, DSPs |
| Conveyor speed range | Minimum and maximum stable speeds |
| Actuator success rate | Correct physical interception rate |
| Jam rate | Jams per test run |
| Host display frame rate | FPS of live dashboard |
| Communication reliability | Packet loss or malformed messages |
| Power consumption | System power under load |

---

## 11. Risk Analysis

| Risk | Impact | Mitigation |
|---|---|---|
| Camera integration with FPGA is difficult | High | Start with USB camera prototype; choose board with known camera support |
| FPGA video pipeline takes longer than expected | High | Implement simple thresholding first, then add morphology and CCL |
| Items overlap too much | Medium | Use passive chute, rails, spacing gate, or lower feed rate |
| Actuator misses items | High | Use encoder tracking and calibrate actuator delay |
| Full CNN on FPGA is too difficult | High | Use feature-based classifier as baseline |
| USB video bandwidth is insufficient | Medium | Use Ethernet or send only processed low-resolution frames |
| Conveyor speed varies | Medium | Use encoder feedback instead of time-only tracking |
| Lighting causes unstable detection | Medium | Add LED panel, diffuser, and fixed exposure |
| Mechanical jams occur | Medium | Use wider belt, rails, and reject logic |
| Team integration happens too late | High | Define interface packets early and test modules with simulators |

---

## 12. Proposed Milestones

## Milestone 1: Requirements and Architecture

Deliverables:
- Project requirements
- System block diagram
- Initial bill of materials
- Interface definitions
- Risk register
- Test item set selected

Exit criteria:
- Team agrees on baseline architecture
- FPGA board, camera, actuator, and conveyor strategy selected

---

## Milestone 2: Vision Prototype on Laptop

Deliverables:
- USB camera test rig
- OpenCV detection prototype
- Dataset capture script
- Initial classification method
- Baseline accuracy results

Exit criteria:
- Laptop can detect and classify static or slow-moving items
- Raw and processed video can be displayed

---

## Milestone 3: Conveyor and Actuator Prototype

Deliverables:
- Conveyor assembled
- Motor driver working
- Encoder feedback working
- Basic actuator mounted
- Manual actuator test

Exit criteria:
- Conveyor runs at controlled speed
- Actuator can physically divert or pick items
- Encoder count correlates with belt movement

---

## Milestone 4: FPGA Vision Pipeline

Deliverables:
- HDL camera or frame input module
- Thresholding module
- Morphology module
- Connected-component labeling module
- Bounding-box output
- Simulation testbenches

Exit criteria:
- FPGA outputs object centroid and bounding box from test frames
- Pipeline latency is measurable

---

## Milestone 5: Host Dashboard and Communication

Deliverables:
- Laptop GUI
- USB/Ethernet communication
- Live video display
- Detection overlays
- Telemetry logging
- Configuration controls

Exit criteria:
- Laptop receives detection metadata
- Dashboard shows bounding boxes and system status
- Logs are saved for later analysis

---

## Milestone 6: Tracking and Actuation Scheduling

Deliverables:
- Encoder-based object tracking
- Camera-to-actuator calibration
- Actuator delay calibration
- Actuation queue
- Conflict handling policy

Exit criteria:
- Single moving item can be detected and sorted reliably
- Actuation command is triggered using encoder position

---

## Milestone 7: Full System Integration

Deliverables:
- Conveyor + vision + tracking + actuator integration
- Multi-item test mode
- Random dump test mode
- Safety stop and fault handling
- Full logging

Exit criteria:
- Multiple items can be sorted in a complete system run
- System can recover from missed detections or uncertain items

---

## Milestone 8: Validation and Final Demo

Deliverables:
- Accuracy test results
- Throughput test results
- Latency measurements
- FPGA resource utilization report
- Demo script
- Final report
- Final presentation

Exit criteria:
- System demonstrates repeatable sorting
- Quantitative results are documented
- Capstone novelty is clearly defended

---

## 13. Timeline Roadmap

This roadmap assumes a two-semester capstone schedule of approximately 24 weeks. Compress or expand as needed.

---

## Phase 1: Project Definition and Feasibility  
**Weeks 1–2**

### Tasks

- Finalize item categories.
- Select baseline sorting mechanism.
- Select FPGA board.
- Select camera architecture.
- Select conveyor architecture.
- Define technical novelty.
- Define success metrics.
- Create high-level block diagram.
- Draft system requirements.

### Deliverables

- Requirements document
- Initial architecture diagram
- Bill of materials draft
- Risk register
- Interface planning document

### Target Decision

Choose one baseline architecture:

```text
SoC FPGA + camera input + laptop dashboard + encoder-synchronized diverter
```

---

## Phase 2: Laptop CV Prototype and Dataset Setup  
**Weeks 3–5**

### Tasks

- Build simple camera test setup.
- Write OpenCV capture script.
- Collect initial dataset.
- Implement thresholding and segmentation.
- Test contour detection.
- Extract shape/color features.
- Train or tune baseline classifier.
- Build simple video overlay display.

### Deliverables

- Dataset v1
- OpenCV prototype
- Static classification accuracy report
- Initial GUI or debug display
- Selected feature set or model architecture

### Exit Criteria

- Laptop can detect and classify selected items under controlled lighting.
- Detection results include bounding boxes, centroids, and labels.

---

## Phase 3: Conveyor, Motor, Encoder, and Actuator Prototype  
**Weeks 4–7**

### Tasks

- Assemble conveyor frame.
- Mount motor and belt.
- Integrate motor driver.
- Read encoder feedback.
- Implement speed control.
- Mount actuator.
- Test servo, solenoid, or magnet.
- Measure actuator response delay.
- Design output bins.

### Deliverables

- Working conveyor prototype
- Encoder calibration
- Actuator test results
- Mechanical CAD updates
- Initial safety checklist

### Exit Criteria

- Conveyor moves items smoothly.
- Encoder counts can estimate belt displacement.
- Actuator can physically redirect or pick an item.

---

## Phase 4: FPGA Input and Basic Image Pipeline  
**Weeks 6–10**

### Tasks

- Set up FPGA toolchain.
- Create or import camera/frame input module.
- Implement pixel stream pipeline.
- Implement grayscale or HSV conversion.
- Implement thresholding.
- Simulate modules with test images.
- Verify output against OpenCV reference.
- Send processed results to embedded processor or host.

### Deliverables

- HDL thresholding pipeline
- Simulation testbench
- OpenCV-vs-HDL comparison
- Basic FPGA image output or metadata output

### Exit Criteria

- FPGA can process image frames or image streams.
- FPGA output matches expected segmentation behavior.

---

## Phase 5: FPGA Object Localization  
**Weeks 10–14**

### Tasks

- Implement morphology filter.
- Implement connected-component labeling or blob detection.
- Extract bounding boxes.
- Extract centroids.
- Extract area and shape features.
- Add frame/object metadata registers.
- Validate with recorded test frames.
- Measure FPGA latency and resource usage.

### Deliverables

- HDL morphology module
- HDL connected-component or blob module
- Bounding-box extraction module
- Object metadata interface
- FPGA utilization report v1

### Exit Criteria

- FPGA outputs object bounding boxes and centroids in real time.
- Laptop or embedded processor can read object metadata.

---

## Phase 6: Host Dashboard and Communication Integration  
**Weeks 12–16**

### Tasks

- Implement USB serial or Ethernet communication.
- Define packet protocol.
- Build host dashboard.
- Display raw or processed video.
- Overlay FPGA detection results.
- Log object metadata.
- Add configuration controls.
- Add calibration interface.

### Deliverables

- Communication protocol document
- Host dashboard v1
- Live overlay display
- CSV or SQLite logging
- Configuration command interface

### Exit Criteria

- Laptop can show video and detection overlays.
- Laptop can log frame/object/actuator data.
- Laptop can send basic commands to embedded system.

---

## Phase 7: Tracking, Scheduling, and Closed-Loop Sorting  
**Weeks 15–19**

### Tasks

- Calibrate pixel-to-mm mapping.
- Calibrate camera-to-actuator distance.
- Calibrate encoder counts per mm.
- Calibrate actuator delay.
- Implement object tracking.
- Implement actuation queue.
- Implement class-to-bin mapping.
- Implement reject logic.
- Test single-item sorting.
- Test multi-item spacing constraints.

### Deliverables

- Calibration results
- Tracking module
- Actuation scheduler
- Single-item sorting demo
- Timing accuracy report

### Exit Criteria

- System can sort one moving object reliably.
- Actuation is triggered by encoder-based position prediction.

---

## Phase 8: Full System Integration and Random Dump Testing  
**Weeks 19–21**

### Tasks

- Integrate feeder/chute/hopper.
- Run randomized item tests.
- Tune conveyor speed.
- Tune detection thresholds.
- Tune classifier confidence thresholds.
- Tune actuator timing.
- Add conflict handling.
- Add jam or fault detection.
- Improve mechanical reliability.

### Deliverables

- Full system prototype
- Random dump test logs
- Sorting accuracy v1
- Throughput estimate
- Reliability issue list

### Exit Criteria

- System can complete repeated sorting runs using randomly placed or randomly fed items.
- Sorting performance is measurable and repeatable.

---

## Phase 9: Validation, Benchmarking, and Optimization  
**Weeks 21–23**

### Tasks

- Run formal test matrix.
- Measure detection accuracy.
- Measure classification accuracy.
- Measure sorting accuracy.
- Measure throughput.
- Measure latency.
- Measure FPGA resource usage.
- Compare FPGA vs CPU processing where possible.
- Document failure cases.
- Optimize bottlenecks.

### Deliverables

- Final validation dataset
- Accuracy plots/tables
- Latency report
- FPGA utilization report
- Throughput report
- Failure analysis

### Exit Criteria

- Final system performance is quantitatively documented.
- Capstone contribution is defensible.

---

## Phase 10: Final Demo, Report, and Presentation  
**Week 24**

### Tasks

- Prepare final demo sequence.
- Prepare backup video.
- Freeze hardware and software.
- Clean up repository.
- Finalize documentation.
- Prepare poster or slide deck.
- Rehearse demo.
- Prepare answers for technical questions.

### Deliverables

- Final prototype
- Final report
- Final presentation
- Demo video
- User guide
- Source code repository
- Bill of materials

### Exit Criteria

- Project can be demonstrated reliably.
- Report explains design, implementation, results, limitations, and future work.

---

## 14. Suggested Gantt-Style Summary

| Phase | Weeks | Main Focus |
|---|---:|---|
| Project definition | 1–2 | Requirements, architecture, BOM |
| Laptop CV prototype | 3–5 | OpenCV, dataset, classifier |
| Conveyor and actuator | 4–7 | Mechanical system, motor, encoder |
| FPGA basic pipeline | 6–10 | Camera/frame input, thresholding |
| FPGA localization | 10–14 | Morphology, connected components, bounding boxes |
| Host dashboard | 12–16 | Video, overlays, USB/Ethernet, logging |
| Tracking and scheduling | 15–19 | Encoder tracking, actuation timing |
| Full integration | 19–21 | Random item sorting |
| Validation | 21–23 | Accuracy, latency, throughput |
| Final demo | 24 | Report, presentation, demonstration |

---

## 15. Minimum Viable Product

The minimum successful system should include:

1. Conveyor belt with controlled speed.
2. Camera with controlled lighting.
3. FPGA-based image segmentation.
4. FPGA-generated bounding boxes and centroids.
5. Host laptop display with detection overlays.
6. Encoder-based tracking from camera to actuator.
7. Servo or solenoid diverter.
8. At least two sorted item categories plus reject handling.
9. Quantitative measurement of sorting accuracy and latency.

---

## 16. Strong Final Deliverable

A strong final system should include:

1. Random item intake using a chute or hopper.
2. FPGA streaming vision pipeline.
3. Connected-component labeling in HDL.
4. Real-time multi-object tracking.
5. Classifier using FPGA-extracted features.
6. Encoder-synchronized actuation.
7. Host dashboard with live video, overlays, logs, and controls.
8. Three or more item categories.
9. Formal FPGA-vs-CPU benchmark.
10. Full validation across speed, lighting, and item-density conditions.

---

## 17. Stretch Features

Possible stretch features include:

- Quantized CNN inference on FPGA
- Electromagnet pickup for ferromagnetic objects
- Multi-actuator sorting
- Reject-bin actuator
- Automatic calibration
- Conveyor speed adaptation based on item density
- Web-based dashboard
- Real-time debug frame streaming from FPGA
- Closed-loop verification sensor after sorting
- Error recovery for missed or uncertain detections

---

## 18. Suggested Team Roles

For a four-person team:

### FPGA / HDL Lead

- Camera interface
- Streaming image pipeline
- Connected-component labeling
- FPGA simulation
- FPGA utilization and timing reports

### Computer Vision / ML Lead

- Dataset collection
- OpenCV reference implementation
- Classifier design
- Calibration algorithms
- Accuracy evaluation

### Mechanical / Actuation Lead

- Conveyor frame
- Feeder or hopper
- Actuator mechanism
- Output bins
- Mechanical reliability and safety

### Embedded / Integration Lead

- Motor control
- Encoder tracking
- Actuator scheduling
- Communication protocol
- Host dashboard integration
- System state machine

---

## 19. Example Demo Script

1. Power on system.
2. Open host laptop dashboard.
3. Show live camera feed.
4. Start conveyor at low speed.
5. Place one item on the belt.
6. Show bounding box and label overlay.
7. Show predicted actuator trigger.
8. Actuator sorts the item into correct bin.
9. Repeat with multiple item categories.
10. Run random dump test.
11. Show dashboard statistics:
    - items detected
    - items classified
    - items sorted correctly
    - latency
    - throughput
12. Show FPGA utilization and CPU comparison.

---

## 20. Final Report Structure

Recommended final report sections:

1. Introduction
2. Problem statement
3. System requirements
4. Related work
5. System architecture
6. Mechanical design
7. Electrical design
8. FPGA vision pipeline
9. Computer vision and classification
10. Communication and host dashboard
11. Tracking and actuation scheduling
12. Calibration
13. Testing methodology
14. Results
15. Risk analysis
16. Discussion
17. Limitations
18. Future work
19. Conclusion
20. Appendix:
    - Schematics
    - HDL module diagrams
    - Protocol definitions
    - BOM
    - Test data
    - Source code references

---

## 21. Final Recommendation

The recommended capstone scope is:

```text
SoC FPGA + direct or supported camera input
+ FPGA HDL vision pipeline
+ host laptop dashboard over USB/Ethernet
+ encoder-synchronized conveyor
+ servo/solenoid diverter actuator
+ quantitative validation
```

This architecture is sufficiently difficult for a 4th-year capstone because it combines:

- FPGA/HDL design
- Real-time computer vision
- Embedded control
- Mechanical design
- Power electronics
- Communication protocols
- Host software
- Calibration and system validation

It is also feasible because the riskiest subsystems can be developed incrementally using a laptop OpenCV prototype, a simple conveyor prototype, and progressively more advanced FPGA acceleration.
