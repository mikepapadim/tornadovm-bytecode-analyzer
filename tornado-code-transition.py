import streamlit as st
import re
import networkx as nx
import graphviz
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from pygments import highlight
from pygments.lexers import JavaLexer, CLexer, CudaLexer
from pygments.formatters import HtmlFormatter
import plotly.graph_objects as go
import sqlite3
from datetime import datetime
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import pandas as pd
from diff_match_patch import diff_match_patch

# Data classes for code representation
@dataclass
class CodeBlock:
    """Represents a block of code with its metadata"""
    code: str
    line_start: int
    line_end: int
    block_type: str  # 'java', 'opencl', or 'ptx'
    labels: List[str] = field(default_factory=list)
    mapped_blocks: List[str] = field(default_factory=list)
    performance_metrics: Dict = field(default_factory=dict)
    method_name: str = ""
    signature: str = ""
    variables: Set[str] = field(default_factory=set)

@dataclass
class CodeMapping:
    """Represents mapping between Java and PTX/OpenCL code blocks"""
    java_block: CodeBlock
    target_blocks: List[CodeBlock]
    mapping_type: str  # 'direct', 'partial', or 'complex'
    performance_metrics: Dict = field(default_factory=dict)
    similarity_score: float = 0.0
    line_mappings: List[Tuple[int, int]] = field(default_factory=list)

class CodeChangeMonitor(FileSystemEventHandler):
    """Monitor code file changes for real-time updates"""
    def __init__(self, callback):
        self.callback = callback
        
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(('.java', '.cl', '.ptx')):
            self.callback(event.src_path)

class TornadoCodeVisualizer:
    """Main class for visualizing code transitions"""
    
    def __init__(self):
        self.java_code: Optional[str] = None
        self.opencl_code: Optional[str] = None
        self.ptx_code: Optional[str] = None
        self.code_mappings: List[CodeMapping] = []
        self.diff_tool = diff_match_patch()
        self.register_usage: Dict[str, List[str]] = {}
        self.memory_patterns: Dict[str, List[str]] = {}
        self.db_conn = self._init_database()
        
    def _init_database(self) -> sqlite3.Connection:
        """Initialize SQLite database for performance tracking"""
        conn = sqlite3.connect(':memory:')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS performance_metrics
                    (timestamp TEXT, metric_type TEXT, metric_name TEXT, value REAL)''')
        conn.commit()
        return conn

    def parse_java_code(self, code: str) -> List[CodeBlock]:
        """Parse Java code into blocks with enhanced metadata"""
        blocks = []
        lines = code.split('\n')
        current_block = []
        start_line = 1
        
        for i, line in enumerate(lines, 1):
            # Look for method declarations and annotations
            if (line.strip().startswith('@Parallel') or 
                line.strip().startswith('public') or 
                line.strip().startswith('private')):
                
                if current_block:
                    # Process the completed block
                    block = self._process_java_block(current_block, start_line, i-1)
                    blocks.append(block)
                
                current_block = [line]
                start_line = i
            else:
                current_block.append(line)
        
        # Process the last block
        if current_block:
            block = self._process_java_block(current_block, start_line, len(lines))
            blocks.append(block)
        
        return blocks

    def _process_java_block(self, lines: List[str], start_line: int, end_line: int) -> CodeBlock:
        """Process a Java code block to extract detailed information"""
        code = '\n'.join(lines)
        
        # Extract annotations
        annotations = [l.strip() for l in lines if '@' in l]
        
        # Extract method name and signature
        method_info = self._extract_method_info(code)
        
        # Extract variables
        variables = self._extract_variables(code)
        
        return CodeBlock(
            code=code,
            line_start=start_line,
            line_end=end_line,
            block_type='java',
            labels=annotations,
            method_name=method_info['name'],
            signature=method_info['signature'],
            variables=variables
        )

    def _extract_method_info(self, code: str) -> Dict[str, str]:
        """Extract method name and signature from Java code"""
        # Match method declaration
        method_pattern = r'(?:public|private|protected)?\s+(?:static\s+)?(\w+)\s+(\w+)\s*\((.*?)\)'
        match = re.search(method_pattern, code, re.MULTILINE | re.DOTALL)
        
        if match:
            return {
                'name': match.group(2),
                'signature': match.group(0).strip(),
                'return_type': match.group(1),
                'parameters': match.group(3).strip()
            }
        return {'name': '', 'signature': '', 'return_type': '', 'parameters': ''}

    def _extract_variables(self, code: str) -> Set[str]:
        """Extract variable declarations and usages from code"""
        variables = set()
        
        # Match variable declarations
        var_pattern = r'\b(?:int|float|double|long|boolean|byte|char|short)\s+(\w+)\s*[=;]'
        variables.update(re.findall(var_pattern, code))
        
        # Match array declarations
        array_pattern = r'\b(?:int|float|double|long|boolean|byte|char|short)\[\]\s+(\w+)\s*[=;]'
        variables.update(re.findall(array_pattern, code))
        
        return variables

    def parse_opencl_code(self, code: str) -> List[CodeBlock]:
        """Parse OpenCL code into blocks with memory access patterns"""
        blocks = []
        lines = code.split('\n')
        current_block = []
        start_line = 1
        
        for i, line in enumerate(lines, 1):
            if line.strip().startswith('__kernel') or line.strip().startswith('// BLOCK'):
                if current_block:
                    blocks.append(CodeBlock(
                        code='\n'.join(current_block),
                        line_start=start_line,
                        line_end=i-1,
                        block_type='opencl'
                    ))
                current_block = [line]
                start_line = i
            else:
                current_block.append(line)
        
        if current_block:
            blocks.append(CodeBlock(
                code='\n'.join(current_block),
                line_start=start_line,
                line_end=len(lines),
                block_type='opencl'
            ))
        
        return blocks

    def parse_ptx_code(self, code: str) -> List[CodeBlock]:
        """Parse PTX code into blocks with register tracking"""
        blocks = []
        lines = code.split('\n')
        current_block = []
        start_line = 1
        
        for i, line in enumerate(lines, 1):
            if line.strip().startswith('.visible .entry') or line.strip().startswith('.func'):
                if current_block:
                    blocks.append(CodeBlock(
                        code='\n'.join(current_block),
                        line_start=start_line,
                        line_end=i-1,
                        block_type='ptx'
                    ))
                current_block = [line]
                start_line = i
            else:
                current_block.append(line)
        
        if current_block:
            blocks.append(CodeBlock(
                code='\n'.join(current_block),
                line_start=start_line,
                line_end=len(lines),
                block_type='ptx'
            ))
        
        return blocks

    def analyze_memory_patterns(self, code: str) -> Dict[str, List[str]]:
        """Analyze OpenCL memory access patterns"""
        patterns = {
            'global_reads': [],
            'global_writes': [],
            'local_access': [],
            'private_access': []
        }
        
        lines = code.split('\n')
        for line in lines:
            if '__global' in line:
                if '*' in line and '=' in line:
                    patterns['global_writes'].append(line.strip())
                else:
                    patterns['global_reads'].append(line.strip())
            elif '__local' in line:
                patterns['local_access'].append(line.strip())
            elif 'private' in line:
                patterns['private_access'].append(line.strip())
                
        return patterns

    def analyze_register_usage(self, code: str) -> Dict[str, List[str]]:
        """Analyze PTX register usage and operations"""
        usage = {
            'declarations': [],
            'moves': [],
            'arithmetic': [],
            'memory': []
        }
        
        lines = code.split('\n')
        for line in lines:
            if '.reg' in line:
                usage['declarations'].append(line.strip())
            elif 'mov.' in line:
                usage['moves'].append(line.strip())
            elif any(op in line for op in ['add.', 'sub.', 'mul.', 'div.', 'mad.']):
                usage['arithmetic'].append(line.strip())
            elif any(op in line for op in ['ld.', 'st.', 'atom.']):
                usage['memory'].append(line.strip())
                
        return usage

    def create_control_flow_graph(self, blocks: List[CodeBlock]) -> graphviz.Digraph:
        """Generate an enhanced control flow graph"""
        dot = graphviz.Digraph(comment='Control Flow Graph')
        dot.attr(rankdir='TB')
        
        # Add nodes for each block
        for i, block in enumerate(blocks):
            # Create a more descriptive label based on block content
            if block.block_type == 'java':
                label = self._extract_method_signature(block.code)
            elif block.block_type == 'opencl':
                label = self._extract_kernel_name(block.code)
            else:  # PTX
                label = self._extract_ptx_entry(block.code)
                
            dot.node(f"block_{i}", label)
            
            # Add edges based on control flow patterns
            if i < len(blocks) - 1:
                dot.edge(f"block_{i}", f"block_{i+1}")
            
            # Add conditional edges based on code analysis
            if any(kw in block.code for kw in ['if', 'for', 'while']):
                for j in range(i+1, len(blocks)):
                    if any(kw in blocks[j].code for kw in ['else', 'break', 'continue', '}']):
                        dot.edge(f"block_{i}", f"block_{j}", color="red")
                        break
        
        return dot

    def _extract_method_signature(self, code: str) -> str:
        """Extract method signature from Java code"""
        lines = code.split('\n')
        for line in lines:
            if 'public' in line or 'private' in line:
                # Clean up and format the signature
                sig = line.strip()
                if len(sig) > 50:
                    sig = sig[:47] + "..."
                return sig
        return "Unknown Method"

    def _extract_kernel_name(self, code: str) -> str:
        """Extract kernel name from OpenCL code"""
        lines = code.split('\n')
        for line in lines:
            if '__kernel' in line:
                match = re.search(r'void\s+(\w+)', line)
                if match:
                    return f"Kernel: {match.group(1)}"
        return "Unknown Kernel"

    def _extract_ptx_entry(self, code: str) -> str:
        """Extract entry point name from PTX code"""
        lines = code.split('\n')
        for line in lines:
            if '.visible .entry' in line or '.func' in line:
                match = re.search(r'(\w+)\(', line)
                if match:
                    return f"PTX: {match.group(1)}"
        return "Unknown PTX Entry"

    def highlight_code(self, code: str, code_type: str, line_numbers: bool = True) -> str:
        """Apply enhanced syntax highlighting with line numbers"""
        if code_type == 'java':
            lexer = JavaLexer()
        elif code_type == 'opencl':
            lexer = CLexer()
        elif code_type == 'ptx':
            lexer = CudaLexer()
        else:
            return code
        
        formatter = HtmlFormatter(
            style='monokai',
            linenos=line_numbers,
            cssclass='highlight',
            lineanchors='line',
            wrapcode=True
        )
        
        highlighted = highlight(code, lexer, formatter)
        css = formatter.get_style_defs('.highlight')
        
        # Add custom CSS for better visibility and interaction
        custom_css = """
        .highlight {
            background: #1e1e1e;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .highlight .hll {
            background-color: #3d3d3d;
            cursor: pointer;
        }
        .highlight .linenos {
            color: #666;
            padding-right: 10px;
            user-select: none;
        }
        .highlight pre {
            margin: 0;
            white-space: pre-wrap;
        }
        """
        
        return f"<style>{css}{custom_css}</style>{highlighted}"

    def create_diff_view(self, original: str, modified: str) -> str:
        """Create a diff view between two code versions"""
        diffs = self.diff_tool.diff_main(original, modified)
        self.diff_tool.diff_cleanupSemantic(diffs)
        
        html = ['<div class="diff-view">']
        for op, text in diffs:
            if op == self.diff_tool.DIFF_INSERT:
                html.append(f'<span class="diff-add">{text}</span>')
            elif op == self.diff_tool.DIFF_DELETE:
                html.append(f'<span class="diff-del">{text}</span>')
            else:
                html.append(f'<span class="diff-equal">{text}</span>')
        html.append('</div>')
        
        # Add diff view styling
        css = """
        <style>
        .diff-view {
            font-family: monospace;
            white-space: pre-wrap;
            background: #1e1e1e;
            padding: 10px;
            border-radius: 5px;
        }
        .diff-add {
            background-color: #144212;
            color: #98c379;
        }
        .diff-del {
            background-color: #421212;
            color: #e06c75;
        }
        .diff-equal {
            color: #abb2bf;
        }
        </style>
        """
        
        return css + ''.join(html)

    def create_mapping_between_languages(self, java_blocks: List[CodeBlock], 
                                      target_blocks: List[CodeBlock], 
                                      block_type: str = 'opencl') -> List[CodeMapping]:
        """Create intelligent mappings between Java and target language blocks"""
        mappings = []
        
        for java_block in java_blocks:
            # Skip blocks without method names
            if not java_block.method_name:
                continue
            
            # Find matching blocks in target language
            matching_blocks = []
            for target_block in target_blocks:
                similarity_score = self._calculate_similarity(java_block, target_block)
                if similarity_score > 0.5:  # Threshold for considering blocks as related
                    matching_blocks.append((target_block, similarity_score))
            
            # Sort matching blocks by similarity score
            matching_blocks.sort(key=lambda x: x[1], reverse=True)
            
            if matching_blocks:
                # Create mapping with the best matches
                mapping = CodeMapping(
                    java_block=java_block,
                    target_blocks=[block for block, _ in matching_blocks],
                    mapping_type='direct' if len(matching_blocks) == 1 else 'complex',
                    similarity_score=matching_blocks[0][1],
                    line_mappings=self._create_line_mappings(java_block, matching_blocks[0][0])
                )
                mappings.append(mapping)
        
        return mappings

    def _calculate_similarity(self, java_block: CodeBlock, target_block: CodeBlock) -> float:
        """Calculate similarity score between Java and target blocks"""
        score = 0.0
        
        # Check method name similarity
        if java_block.method_name.lower() in target_block.code.lower():
            score += 0.4
        
        # Check variable usage similarity
        target_vars = self._extract_variables(target_block.code)
        common_vars = java_block.variables.intersection(target_vars)
        if java_block.variables:
            score += 0.3 * (len(common_vars) / len(java_block.variables))
        
        # Check structural similarity (loops, conditionals)
        java_structures = self._extract_control_structures(java_block.code)
        target_structures = self._extract_control_structures(target_block.code)
        if java_structures:
            score += 0.3 * (len(target_structures) / len(java_structures))
        
        return min(score, 1.0)

    def _extract_control_structures(self, code: str) -> List[str]:
        """Extract control structure patterns from code"""
        structures = []
        
        # Look for loops and conditionals
        patterns = {
            'for': r'for\s*\([^)]*\)',
            'while': r'while\s*\([^)]*\)',
            'if': r'if\s*\([^)]*\)',
            'switch': r'switch\s*\([^)]*\)'
        }
        
        for struct_type, pattern in patterns.items():
            matches = re.findall(pattern, code)
            structures.extend(matches)
        
        return structures

    def _create_line_mappings(self, java_block: CodeBlock, target_block: CodeBlock) -> List[Tuple[int, int]]:
        """Create line-by-line mappings between Java and target blocks"""
        mappings = []
        java_lines = java_block.code.split('\n')
        target_lines = target_block.code.split('\n')
        
        for i, java_line in enumerate(java_lines, java_block.line_start):
            java_tokens = set(re.findall(r'\b\w+\b', java_line))
            
            for j, target_line in enumerate(target_lines, target_block.line_start):
                target_tokens = set(re.findall(r'\b\w+\b', target_line))
                
                # If there's significant token overlap, consider lines as mapped
                if len(java_tokens & target_tokens) >= 2:
                    mappings.append((i, j))
        
        return mappings

    def create_interactive_visualization(self, java_blocks: List[CodeBlock], 
                                      target_blocks: List[CodeBlock], 
                                      mappings: List[CodeMapping]) -> str:
        """Create an interactive visualization with code mappings"""
        # Generate unique IDs for blocks
        java_ids = [f"java-block-{i}" for i in range(len(java_blocks))]
        target_ids = [f"target-block-{i}" for i in range(len(target_blocks))]
        
        # Create HTML structure
        html = """
        <style>
        .code-container {
            display: flex;
            gap: 20px;
            position: relative;
            padding: 20px;
            background: #1e1e1e;
            border-radius: 8px;
        }
        .code-panel {
            flex: 1;
            min-width: 0;
        }
        .code-block {
            position: relative;
            margin: 10px 0;
            padding: 10px;
            background: #2d2d2d;
            border-radius: 4px;
            cursor: pointer;
        }
        .code-block:hover {
            background: #3d3d3d;
        }
        .highlight-active {
            background: #264f78 !important;
        }
        .highlight-mapped {
            background: #3c4c3c !important;
        }
        .connection-line {
            position: absolute;
            background: rgba(100, 149, 237, 0.3);
            height: 2px;
            transform-origin: left center;
            pointer-events: none;
            transition: opacity 0.2s;
        }
        </style>
        
        <div class="code-container" id="code-mapping-container">
            <div class="code-panel" id="java-panel">
                <h3>Java Code</h3>
        """
        
        # Add Java blocks
        for block_id, block in zip(java_ids, java_blocks):
            html += f"""
                <div class="code-block" id="{block_id}" 
                     onclick="highlightMapping('{block_id}')"
                     onmouseenter="showConnections('{block_id}')"
                     onmouseleave="hideConnections('{block_id}')">
                    {self.highlight_code(block.code, 'java')}
                </div>
            """
        
        html += """
            </div>
            <div class="code-panel" id="target-panel">
                <h3>Generated Code</h3>
        """
        
        # Add target blocks
        for block_id, block in zip(target_ids, target_blocks):
            html += f"""
                <div class="code-block" id="{block_id}"
                     onclick="highlightMapping('{block_id}')"
                     onmouseenter="showConnections('{block_id}')"
                     onmouseleave="hideConnections('{block_id}')">
                    {self.highlight_code(block.code, block.block_type)}
                </div>
            """
        
        html += """
            </div>
        </div>
        """
        
        # Add JavaScript for interactivity
        html += """
        <script>
        // Mapping data structure
        const mappings = {
        """
        
        # Add mapping data
        for mapping in mappings:
            java_idx = java_blocks.index(mapping.java_block)
            for target_block in mapping.target_blocks:
                target_idx = target_blocks.index(target_block)
                html += f"""
            '{java_ids[java_idx]}': ['{target_ids[target_idx]}'],
            '{target_ids[target_idx]}': ['{java_ids[java_idx]}'],
                """
        
        html += """
        };
        
        function highlightMapping(blockId) {
            // Remove existing highlights
            document.querySelectorAll('.code-block').forEach(block => {
                block.classList.remove('highlight-active', 'highlight-mapped');
            });
            
            // Add new highlights
            const block = document.getElementById(blockId);
            block.classList.add('highlight-active');
            
            const mappedBlocks = mappings[blockId] || [];
            mappedBlocks.forEach(mappedId => {
                const mappedBlock = document.getElementById(mappedId);
                if (mappedBlock) {
                    mappedBlock.classList.add('highlight-mapped');
                }
            });
            
            drawConnections(blockId);
        }
        
        function showConnections(blockId) {
            drawConnections(blockId);
        }
        
        function hideConnections(blockId) {
            document.querySelectorAll('.connection-line').forEach(line => {
                line.style.opacity = '0';
            });
        }
        
        function drawConnections(blockId) {
            // Remove existing connections
            document.querySelectorAll('.connection-line').forEach(line => line.remove());
            
            const sourceBlock = document.getElementById(blockId);
            const mappedBlocks = mappings[blockId] || [];
            
            mappedBlocks.forEach(targetId => {
                const targetBlock = document.getElementById(targetId);
                if (sourceBlock && targetBlock) {
                    drawConnectionLine(sourceBlock, targetBlock);
                }
            });
        }
        
        function drawConnectionLine(source, target) {
            const container = document.getElementById('code-mapping-container');
            const containerRect = container.getBoundingClientRect();
            const sourceRect = source.getBoundingClientRect();
            const targetRect = target.getBoundingClientRect();
            
            const line = document.createElement('div');
            line.className = 'connection-line';
            
            const startX = sourceRect.right - containerRect.left;
            const startY = sourceRect.top + sourceRect.height/2 - containerRect.top;
            const endX = targetRect.left - containerRect.left;
            const endY = targetRect.top + targetRect.height/2 - containerRect.top;
            
            const length = Math.sqrt(Math.pow(endX - startX, 2) + Math.pow(endY - startY, 2));
            const angle = Math.atan2(endY - startY, endX - startX) * 180 / Math.PI;
            
            line.style.width = `${length}px`;
            line.style.left = `${startX}px`;
            line.style.top = `${startY}px`;
            line.style.transform = `rotate(${angle}deg)`;
            
            container.appendChild(line);
        }
        
        // Initialize the visualization
        document.addEventListener('DOMContentLoaded', function() {
            // Draw initial connections for the first block
            const firstBlock = document.querySelector('.code-block');
            if (firstBlock) {
                highlightMapping(firstBlock.id);
            }
        });
        </script>
        """
        
        return html

    def analyze_performance_metrics(self, code_blocks: List[CodeBlock], block_type: str) -> None:
        """Analyze performance metrics for code blocks"""
        for block in code_blocks:
            metrics = {}
            
            if block_type == 'opencl':
                # Analyze memory access patterns
                patterns = self.analyze_memory_patterns(block.code)
                metrics['memory_patterns'] = patterns
                
                # Calculate memory intensity
                total_memory_ops = (len(patterns['global_reads']) + 
                                  len(patterns['global_writes']) +
                                  len(patterns['local_access']))
                metrics['memory_intensity'] = total_memory_ops / len(block.code.split('\n'))
                
                # Analyze thread divergence potential
                metrics['thread_divergence'] = self._analyze_thread_divergence(block.code)
                
            elif block_type == 'ptx':
                # Analyze register usage
                reg_usage = self.analyze_register_usage(block.code)
                metrics['register_usage'] = reg_usage
                
                # Calculate arithmetic intensity
                total_arithmetic = len(reg_usage['arithmetic'])
                total_memory = len(reg_usage['memory'])
                metrics['arithmetic_intensity'] = (
                    total_arithmetic / total_memory if total_memory > 0 else 0
                )
                
                # Analyze instruction mix
                metrics['instruction_mix'] = self._analyze_instruction_mix(block.code)
            
            # Store metrics in block
            block.performance_metrics = metrics
            
            # Store in database for tracking
            self._store_metrics(block, metrics)

    def _analyze_thread_divergence(self, code: str) -> Dict[str, int]:
        """Analyze potential thread divergence in OpenCL code"""
        divergence = {
            'if_statements': 0,
            'thread_dependent_branches': 0,
            'loop_divergence': 0
        }
        
        lines = code.split('\n')
        for line in lines:
            # Count if statements
            if re.search(r'\bif\s*\(', line):
                divergence['if_statements'] += 1
                # Check if condition depends on thread ID
                if re.search(r'get_(local|global)_id', line):
                    divergence['thread_dependent_branches'] += 1
            
            # Check for divergent loops
            if re.search(r'\bfor\s*\(', line) or re.search(r'\bwhile\s*\(', line):
                if re.search(r'get_(local|global)_id', line):
                    divergence['loop_divergence'] += 1
        
        return divergence

    def _analyze_instruction_mix(self, code: str) -> Dict[str, int]:
        """Analyze PTX instruction mix"""
        instruction_mix = {
            'arithmetic': 0,
            'memory': 0,
            'control': 0,
            'conversion': 0,
            'special': 0
        }
        
        lines = code.split('\n')
        for line in lines:
            if re.search(r'\b(add|sub|mul|div|mad)\.\w+', line):
                instruction_mix['arithmetic'] += 1
            elif re.search(r'\b(ld|st|atom)\.\w+', line):
                instruction_mix['memory'] += 1
            elif re.search(r'\b(bra|ret|call)\b', line):
                instruction_mix['control'] += 1
            elif re.search(r'\b(cvt)\.\w+', line):
                instruction_mix['conversion'] += 1
            elif re.search(r'\b(sin|cos|sqrt|rcp)\.\w+', line):
                instruction_mix['special'] += 1
        
        return instruction_mix

    def _store_metrics(self, block: CodeBlock, metrics: Dict) -> None:
        """Store performance metrics in database"""
        timestamp = datetime.now().isoformat()
        
        # Flatten metrics for storage
        def flatten_metrics(prefix: str, metric_dict: Dict) -> List[Tuple[str, float]]:
            flattened = []
            for key, value in metric_dict.items():
                if isinstance(value, dict):
                    flattened.extend(flatten_metrics(f"{prefix}_{key}", value))
                elif isinstance(value, (int, float)):
                    flattened.append((f"{prefix}_{key}", float(value)))
                elif isinstance(value, list):
                    flattened.append((f"{prefix}_{key}_count", float(len(value))))
            return flattened
        
        # Store flattened metrics
        cursor = self.db_conn.cursor()
        for metric_name, value in flatten_metrics("", metrics):
            cursor.execute(
                "INSERT INTO performance_metrics VALUES (?, ?, ?, ?)",
                (timestamp, block.block_type, metric_name, value)
            )
        self.db_conn.commit()

    def create_performance_visualization(self, blocks: List[CodeBlock]) -> Dict[str, go.Figure]:
        """Create performance visualizations"""
        visualizations = {}
        
        # Memory access pattern visualization for OpenCL
        opencl_blocks = [b for b in blocks if b.block_type == 'opencl']
        if opencl_blocks:
            mem_patterns = []
            for block in opencl_blocks:
                if 'memory_patterns' in block.performance_metrics:
                    patterns = block.performance_metrics['memory_patterns']
                    mem_patterns.append({
                        'block': f"Block {blocks.index(block)}",
                        'global_reads': len(patterns['global_reads']),
                        'global_writes': len(patterns['global_writes']),
                        'local_access': len(patterns['local_access'])
                    })
            
            if mem_patterns:
                df = pd.DataFrame(mem_patterns)
                fig = go.Figure()
                
                # Add bars for each memory access type
                for col in ['global_reads', 'global_writes', 'local_access']:
                    fig.add_trace(go.Bar(
                        name=col.replace('_', ' ').title(),
                        x=df['block'],
                        y=df[col],
                        text=df[col],
                        textposition='auto',
                    ))
                
                fig.update_layout(
                    title="Memory Access Patterns",
                    barmode='group',
                    xaxis_title="Code Block",
                    yaxis_title="Number of Accesses",
                    template="plotly_dark",
                    height=400
                )
                
                visualizations['memory_patterns'] = fig
        
        # Instruction mix visualization for PTX
        ptx_blocks = [b for b in blocks if b.block_type == 'ptx']
        if ptx_blocks:
            inst_mix = []
            for block in ptx_blocks:
                if 'instruction_mix' in block.performance_metrics:
                    mix = block.performance_metrics['instruction_mix']
                    mix['block'] = f"Block {blocks.index(block)}"
                    inst_mix.append(mix)
            
            if inst_mix:
                df = pd.DataFrame(inst_mix)
                fig = go.Figure()
                
                # Create pie chart for instruction mix
                for block in df['block']:
                    values = df[df['block'] == block].iloc[0][['arithmetic', 'memory', 'control', 'conversion', 'special']]
                    fig.add_trace(go.Pie(
                        labels=['Arithmetic', 'Memory', 'Control', 'Conversion', 'Special'],
                        values=values,
                        name=block,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title=block
                    ))
                
                fig.update_layout(
                    title="PTX Instruction Mix",
                    template="plotly_dark",
                    height=400,
                    showlegend=True
                )
                
                visualizations['instruction_mix'] = fig
        
        return visualizations

    def create_control_flow_visualization(self, blocks: List[CodeBlock]) -> Dict[str, graphviz.Digraph]:
        """Create control flow visualizations for each block type"""
        visualizations = {}
        
        for block_type in ['java', 'opencl', 'ptx']:
            type_blocks = [b for b in blocks if b.block_type == block_type]
            if not type_blocks:
                continue
            
            dot = graphviz.Digraph(comment=f'{block_type.upper()} Control Flow')
            dot.attr(rankdir='TB', 
                    bgcolor='transparent',
                    fontcolor='white',
                    fontname='Arial')
            
            # Node attributes
            dot.attr('node',
                    shape='box',
                    style='filled',
                    fillcolor='#2d2d2d',
                    color='#666666',
                    fontcolor='white',
                    fontname='Arial')
            
            # Edge attributes
            dot.attr('edge',
                    color='#666666',
                    fontcolor='white',
                    fontname='Arial')
            
            # Add nodes and edges
            for i, block in enumerate(type_blocks):
                # Create node label
                if block_type == 'java':
                    label = block.method_name or f"Block {i}"
                elif block_type == 'opencl':
                    label = f"Kernel {i}"
                    if 'memory_patterns' in block.performance_metrics:
                        patterns = block.performance_metrics['memory_patterns']
                        label += f"\nGlobal R/W: {len(patterns['global_reads'])}/{len(patterns['global_writes'])}"
                else:  # PTX
                    label = f"PTX Block {i}"
                    if 'instruction_mix' in block.performance_metrics:
                        mix = block.performance_metrics['instruction_mix']
                        label += f"\nArith/Mem: {mix['arithmetic']}/{mix['memory']}"
                
                # Add node
                node_id = f"{block_type}_{i}"
                dot.node(node_id, label)
                
                # Add edges based on control flow analysis
                if i > 0:
                    prev_id = f"{block_type}_{i-1}"
                    dot.edge(prev_id, node_id)
                
                # Add edges for control structures
                structures = self._extract_control_structures(block.code)
                if structures:
                    # Add a subnode for each control structure
                    for j, struct in enumerate(structures):
                        struct_id = f"{node_id}_struct_{j}"
                        dot.node(struct_id, struct[:30] + "..." if len(struct) > 30 else struct,
                               shape='ellipse',
                               fillcolor='#1e4d2b')
                        dot.edge(node_id, struct_id, style='dashed')
            
            visualizations[block_type] = dot
        
        return visualizations

def main():
    st.set_page_config(
        page_title="TornadoVM Code Transition Visualizer",
        page_icon="🌪️",
        layout="wide"
    )
    
    st.title("🌪️ TornadoVM Code Transition Visualizer")
    
    # Initialize visualizer
    visualizer = TornadoCodeVisualizer()
    
    # File upload section
    with st.sidebar:
        st.header("Upload Code Files")
        java_file = st.file_uploader("Upload Java Source", type=['java', 'txt'])
        opencl_file = st.file_uploader("Upload OpenCL Code", type=['cl', 'txt'])
        ptx_file = st.file_uploader("Upload PTX Code", type=['ptx', 'txt'])
        
        st.header("Display Options")
        show_line_numbers = st.checkbox("Show Line Numbers", value=True)
        show_connections = st.checkbox("Show Code Connections", value=True)
        show_metrics = st.checkbox("Show Performance Metrics", value=True)
    
    # Main content area
    if java_file and (opencl_file or ptx_file):
        # Parse uploaded files
        java_code = java_file.getvalue().decode('utf-8')
        java_blocks = visualizer.parse_java_code(java_code)
        
        opencl_blocks = []
        ptx_blocks = []
        
        if opencl_file:
            opencl_code = opencl_file.getvalue().decode('utf-8')
            opencl_blocks = visualizer.parse_opencl_code(opencl_code)
            # Analyze OpenCL performance
            visualizer.analyze_performance_metrics(opencl_blocks, 'opencl')
            # Create mappings for OpenCL
            opencl_mappings = visualizer.create_mapping_between_languages(java_blocks, opencl_blocks, 'opencl')
            
        if ptx_file:
            ptx_code = ptx_file.getvalue().decode('utf-8')
            ptx_blocks = visualizer.parse_ptx_code(ptx_code)
            # Analyze PTX performance
            visualizer.analyze_performance_metrics(ptx_blocks, 'ptx')
            # Create mappings for PTX
            ptx_mappings = visualizer.create_mapping_between_languages(java_blocks, ptx_blocks, 'ptx')
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Code Mapping", "Control Flow", "Performance"])
        
        with tab1:
            st.subheader("Interactive Code Mapping")
            
            if opencl_blocks:
                st.markdown("### Java ↔ OpenCL Mapping")
                st.components.v1.html(
                    visualizer.create_interactive_visualization(
                        java_blocks, opencl_blocks, opencl_mappings
                    ),
                    height=800,
                    scrolling=True
                )
            
            if ptx_blocks:
                st.markdown("### Java ↔ PTX Mapping")
                st.components.v1.html(
                    visualizer.create_interactive_visualization(
                        java_blocks, ptx_blocks, ptx_mappings
                    ),
                    height=800,
                    scrolling=True
                )
        
        with tab2:
            st.subheader("Control Flow Analysis")
            
            # Create control flow visualizations
            cf_graphs = visualizer.create_control_flow_visualization(
                java_blocks + opencl_blocks + ptx_blocks
            )
            
            # Display control flow graphs
            for lang, graph in cf_graphs.items():
                st.markdown(f"### {lang.upper()} Control Flow")
                st.graphviz_chart(graph, use_container_width=True)
        
        with tab3:
            st.subheader("Performance Analysis")
            
            # Create performance visualizations
            perf_viz = visualizer.create_performance_visualization(
                opencl_blocks + ptx_blocks
            )
            
            # Display performance visualizations
            for viz_name, fig in perf_viz.items():
                st.plotly_chart(fig, use_container_width=True)
            
            # Display detailed metrics
            if show_metrics:
                st.markdown("### Detailed Metrics")
                
                if opencl_blocks:
                    st.markdown("#### OpenCL Metrics")
                    for i, block in enumerate(opencl_blocks):
                        with st.expander(f"Block {i} Metrics"):
                            st.json(block.performance_metrics)
                
                if ptx_blocks:
                    st.markdown("#### PTX Metrics")
                    for i, block in enumerate(ptx_blocks):
                        with st.expander(f"Block {i} Metrics"):
                            st.json(block.performance_metrics)
    
    else:
        st.info("Please upload the Java source code and at least one of OpenCL or PTX code files to begin visualization.")

if __name__ == "__main__":
    main() 