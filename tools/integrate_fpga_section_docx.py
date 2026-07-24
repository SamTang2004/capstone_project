from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor


INPUT = Path(r"C:\Users\20sam\Downloads\Detailed Design.docx")
OUTPUT = Path(r"C:\python_projects\capstone_project\Detailed Design - FPGA Integrated.docx")


def set_run_font(run, size=None, bold=None, italic=None, color=None, font="Calibri"):
    run.font.name = font
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.rFonts
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), font)
    rfonts.set(qn("w:hAnsi"), font)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table, color="BFC9D6"):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = borders.find(qn(f"w:{edge}"))
        if el is None:
            el = OxmlElement(f"w:{edge}")
            borders.append(el)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "6")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), color)


def style_table(table):
    table.autofit = True
    set_table_borders(table)
    for r_idx, row in enumerate(table.rows):
        for cell in row.cells:
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell)
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(0)
                for run in p.runs:
                    set_run_font(run, size=9, bold=(r_idx == 0))
            if r_idx == 0:
                set_cell_shading(cell, "E8EEF5")


def add_paragraph_after(doc, after, text="", style=None, bold=False, italic=False, font="Calibri", size=None):
    p = doc.add_paragraph(style=style)
    if text:
        run = p.add_run(text)
        set_run_font(run, size=size, bold=bold, italic=italic, font=font)
    p.paragraph_format.space_after = Pt(6)
    after.addnext(p._p)
    return p._p


def add_heading_after(doc, after, text, style="Heading 5"):
    p = doc.add_paragraph(style=style)
    p.add_run(text)
    after.addnext(p._p)
    return p._p


def add_bullet_after(doc, after, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Pt(18)
    p.paragraph_format.first_line_indent = Pt(-9)
    p.add_run("- ")
    p.add_run(text)
    p.paragraph_format.space_after = Pt(3)
    after.addnext(p._p)
    return p._p


def add_number_after(doc, after, text):
    p = doc.add_paragraph()
    p.add_run(text)
    p.paragraph_format.space_after = Pt(3)
    after.addnext(p._p)
    return p._p


def add_code_after(doc, after, lines):
    for line in lines:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Pt(12)
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(line)
        set_run_font(run, font="Consolas", size=8.5)
        after.addnext(p._p)
        after = p._p
    return after


def add_table_after(doc, after, headers, rows):
    table = doc.add_table(rows=1, cols=len(headers))
    for idx, header in enumerate(headers):
        table.cell(0, idx).text = header
    for row in rows:
        cells = table.add_row().cells
        for idx, value in enumerate(row):
            cells[idx].text = value
    style_table(table)
    after.addnext(table._tbl)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(3)
    table._tbl.addnext(spacer._p)
    return spacer._p


def remove_between(start_el, end_el):
    body = start_el.getparent()
    deleting = False
    for child in list(body):
        if child is start_el:
            deleting = True
            continue
        if child is end_el:
            break
        if deleting:
            body.remove(child)


def find_section_bounds(doc):
    start = end = None
    for p in doc.paragraphs:
        text = p.text.strip()
        if text == "3.2 FPGA Processing Subsystem":
            start = p
        elif text == "3.3 Control and Actuation Subsystem":
            end = p
            break
    if start is None or end is None:
        raise RuntimeError("Could not find section 3.2 / 3.3 boundaries")
    return start, end


def integrate():
    doc = Document(INPUT)
    start, end = find_section_bounds(doc)
    remove_between(start._p, end._p)
    cur = start._p

    cur = add_paragraph_after(
        doc,
        cur,
        "The FPGA processing subsystem is the hardware-accelerated vision and inference engine for VisionSort. "
        "Its final design uses the Zynq programmable logic (PL) for deterministic INT8 neural-network acceleration "
        "and the Zynq processing system (PS) for runtime control, buffer management, and graph scheduling. "
        "A host laptop prepares the quantized model, uploads model buffers, and displays system results, while the "
        "FPGA board performs the time-critical inspection computation close to the camera and conveyor control path.",
    )
    cur = add_paragraph_after(
        doc,
        cur,
        "The final design is divided into three coordinated layers: a Python compiler/runtime on the host, C firmware "
        "on the Zynq PS, and SystemVerilog RTL in the Zynq PL. This split keeps model preparation and debugging in "
        "software while reserving the PL for regular high-throughput operations such as matrix multiplication and "
        "requantization.",
    )

    cur = add_heading_after(doc, cur, "3.2.1 Final FPGA Architecture")
    cur = add_code_after(
        doc,
        cur,
        [
            "Host laptop: Python API, quantization, model package builder, dashboard",
            "        |",
            "        | ALLOC / WRITE / READ / EXECUTE / FREE",
            "        v",
            "Zynq PS: protocol server, buffer manager, graph executor, PL driver",
            "        |",
            "        | AXI-Lite control + shared DDR buffers",
            "        v",
            "Zynq PL: tiled INT8 GEMM, INT32 accumulation, INT8 requantization",
        ],
    )
    cur = add_paragraph_after(
        doc,
        cur,
        "The host never receives physical DDR addresses. Instead, it receives logical buffer IDs. The PS translates "
        "those IDs into physical addresses before programming the PL. This makes the software interface safer and "
        "keeps board-specific memory details inside the embedded runtime.",
    )

    cur = add_heading_after(doc, cur, "3.2.2 Host Compiler and Model Packaging")
    cur = add_paragraph_after(
        doc,
        cur,
        "The host compiler converts the inspection CNN into a compact resident model package. Weights are quantized "
        "to INT8, layer metadata is serialized into a flat descriptor, and all model buffers are uploaded once before "
        "inspection begins. During operation, the host sends only the input tensor and reads back the final result.",
    )
    cur = add_table_after(
        doc,
        cur,
        ["Structure", "Purpose"],
        [
            ["BcudaModelDesc", "Model magic, version, layer count, buffer-table offset, layer-table offset, and input/output indexes."],
            ["BcudaModelBufferDesc[]", "Maps graph buffer indexes to device buffer IDs, dtype, layout, usage, size, and shape metadata."],
            ["BcudaLayerDesc[]", "Stores operation type, input/output indexes, weights, bias/scale buffers, and layer dimensions."],
        ],
    )
    cur = add_paragraph_after(
        doc,
        cur,
        "The supported graph operations are GEMM, convolution lowered to GEMM, activation, requantization, pooling, "
        "upsampling, channel concatenation, reshape, detection decode, and non-maximum suppression. Dense arithmetic "
        "is assigned to the PL; smaller or branch-heavy operations remain on the PS.",
    )

    cur = add_heading_after(doc, cur, "3.2.3 Firmware Runtime")
    cur = add_paragraph_after(
        doc,
        cur,
        "The firmware is the local device runtime. It validates protocol packets, allocates DDR buffers, tracks buffer "
        "IDs, parses model descriptors, dispatches graph layers, and programs the PL accelerator through AXI-Lite "
        "registers. The external protocol intentionally uses only five commands:",
    )
    cur = add_table_after(
        doc,
        cur,
        ["Command", "Role"],
        [
            ["ALLOC", "Create a typed device buffer and return a logical buffer ID."],
            ["WRITE", "Copy host data into a device buffer."],
            ["READ", "Copy final result data back to the host."],
            ["EXECUTE", "Run a GEMM, graph, or operation descriptor."],
            ["FREE", "Release a device buffer."],
        ],
    )
    cur = add_paragraph_after(
        doc,
        cur,
        "Keeping the command set small avoids a separate transport command for every possible neural-network layer. "
        "Layer-specific behavior is represented inside EXECUTE descriptors instead.",
    )

    cur = add_heading_after(doc, cur, "3.2.4 RTL-Level Accelerator")
    cur = add_paragraph_after(
        doc,
        cur,
        "The PL accelerator contains AXI-Lite control registers, a GEMM controller, tile loader, memory IO driver, "
        "16x16 INT8 systolic array group, accumulator buffer, tile writer, and INT32-to-INT8 requantizer. The primary "
        "datapath multiplies signed INT8 activations by signed INT8 weights and accumulates into signed INT32 values.",
    )
    cur = add_code_after(
        doc,
        cur,
        [
            "for each output tile row:",
            "  for each output tile column group:",
            "    clear INT32 accumulators",
            "    for each K tile:",
            "      load A and B INT8 tiles from DDR",
            "      run 16x16 systolic compute",
            "      accumulate partial sums",
            "    write output tile or requantized INT8 activation",
        ],
    )
    cur = add_paragraph_after(
        doc,
        cur,
        "The memory IO driver converts unaligned INT8 byte reads and aligned INT32 writes into 32-bit DDR commands. "
        "Write requests receive priority over reads when both are ready, preventing completed output tiles from "
        "stalling during writeback.",
    )

    cur = add_heading_after(doc, cur, "3.2.5 Register Interface")
    cur = add_table_after(
        doc,
        cur,
        ["Offset", "Register", "Purpose"],
        [
            ["0x00", "CONTROL", "Start bit and operation-mode bit."],
            ["0x04", "STATUS", "Busy, done, and error status."],
            ["0x08-0x10", "A_BASE/B_BASE/C_BASE", "DDR base addresses for input, weight, and output buffers."],
            ["0x14-0x1c", "A_STRIDE/B_STRIDE/C_STRIDE", "Row strides in bytes."],
            ["0x20-0x28", "M/N/K", "Matrix dimensions or requantization element count."],
            ["0x2c-0x30", "TILE_SIZE/ARRAY_COUNT", "Tile size and number of active 16x16 array lanes."],
            ["0x34-0x3c", "RQ_*", "Fixed-point requantization multiplier, shift, and output zero point."],
        ],
    )

    cur = add_heading_after(doc, cur, "3.2.6 Quantitative Technical Analysis")
    cur = add_paragraph_after(
        doc,
        cur,
        "A 16x16 systolic array has 256 processing elements and performs 256 MACs per cycle. At a 100 MHz PL clock, "
        "one array provides 25.6 GMAC/s, or 51.2 GOPS when multiply and add are counted separately. The scalable final "
        "architecture can enable up to eight 16x16 arrays, giving 2048 MACs/cycle, 204.8 GMAC/s, or 409.6 GOPS at "
        "100 MHz before memory and control overhead.",
    )
    cur = add_paragraph_after(
        doc,
        cur,
        "INT32 accumulation is selected because the worst-case signed INT8 product magnitude is 128 x 128 = 16,384. "
        "A signed INT32 accumulator can therefore support K = floor(2,147,483,647 / 16,384) = 131,071 worst-case "
        "products, far above the K dimensions expected in the inspection CNN. INT16 accumulation was rejected because "
        "it would overflow for realistic layers.",
    )
    cur = add_paragraph_after(
        doc,
        cur,
        "For one 16x16x16 tile, the array performs 4096 MACs. The compulsory traffic is 256 bytes for A, 256 bytes for "
        "B, and 1024 bytes for an INT32 C tile, or 1536 bytes total. The arithmetic intensity is therefore 4096 / "
        "1536 = 2.67 MAC/byte, and it improves for larger K because C is written only after multiple K tiles are "
        "accumulated. Requantization reduces intermediate activation storage from 4E bytes to E bytes for E elements, "
        "a 4:1 bandwidth reduction.",
    )

    cur = add_heading_after(doc, cur, "3.2.7 Design Iterations and Justification")
    cur = add_table_after(
        doc,
        cur,
        ["Design question", "Final decision and reason"],
        [
            ["Protocol design", "Use five generic commands instead of many layer-specific commands; this keeps the host/PS interface stable."],
            ["Memory model", "Use logical buffer IDs instead of host-visible physical addresses; this improves safety and portability."],
            ["Numeric format", "Use INT8 inputs/weights with INT32 accumulation; this fits Zynq resources and protects against overflow."],
            ["Hardware primitive", "Use GEMM as the main accelerator primitive; it supports both MLP layers and convolution lowering."],
            ["Operator placement", "Run dense arithmetic in PL and irregular operations on PS; this keeps RTL manageable."],
            ["Activation format", "Requantize INT32 outputs to INT8; this reduces activation traffic by 4x for multi-layer CNNs."],
        ],
    )

    cur = add_heading_after(doc, cur, "3.2.8 How the FPGA Design Satisfies the Specifications")
    for item in [
        "FS1 and FS2 are supported because the FPGA subsystem accelerates CNN inference used to classify inspected food items.",
        "FS3 is supported because the model output is produced as a compact class/quality result that the actuation subsystem can schedule.",
        "The low-latency requirement is addressed by keeping model weights resident in DDR and avoiding host round trips between layers.",
        "The reliability requirement is addressed with deterministic graph descriptors, fixed register maps, range checks, and golden-trace RTL simulations.",
        "The prototype requirement is addressed because another group can construct the subsystem from the documented host package, firmware, RTL modules, register map, and Vivado scripts.",
    ]:
        cur = add_bullet_after(doc, cur, item)

    cur = add_heading_after(doc, cur, "3.2.9 Hardware and Code References")
    cur = add_paragraph_after(
        doc,
        cur,
        "Hardware references used for this subsystem are AMD Zynq-7000 Overview DS190, AMD Zynq-7010/Zynq-7020 "
        "Electrical Data Sheet DS187, AMD Zynq-7000 Technical Reference Manual UG585, AMD AXI Reference Guide UG1037, "
        "and the TUL PYNQ-Z2 reference manual/product page. Project code is custom and is not copied from external "
        "repositories. Internal references include FPGA/budgetcuda/compiler.py, FPGA/firmware/include/bcuda_protocol.h, "
        "FPGA/firmware/src/graph_executor.c, FPGA/firmware/src/axi_dma.c, FPGA/hardware/rtl/bcuda_tiled_systolic_ddr_top.sv, "
        "FPGA/hardware/rtl/gemm_controller.sv, FPGA/hardware/rtl/systolic_array_16x16.sv, and FPGA/hardware/docs/bcuda_ddr_gemm_register_map.md.",
    )

    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    integrate()
