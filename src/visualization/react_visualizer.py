import streamlit as st
import streamlit.components.v1 as components
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import re
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

@dataclass
class CodeBlock:
    """Represents a block of code with metadata"""
    id: str
    code: str
    language: str
    line_start: int
    line_end: int
    block_type: str
    metrics: Optional[Dict] = None

@dataclass
class LineMapping:
    """Represents a mapping between source and target code lines"""
    source_lines: List[int]
    target_lines: List[int]
    type: str
    description: str

@dataclass
class CodeMapping:
    """Represents a mapping between source and target code blocks"""
    source_id: str
    target_id: str
    type: str
    confidence: float
    line_mapping: List[LineMapping]

class ReactVisualizer:
    """Handles visualization using Streamlit components"""
    
    def __init__(self):
        self.java_patterns = {}
        self.target_patterns = {}
        self.target_type = "ptx"  # Default to PTX
        self.mapping_stats = {
            'total_mappings': 0,
            'confidence': {'high': 0, 'medium': 0, 'low': 0},
            'by_type': {},
            'by_subtype': {}
        }
    
    def display_code_transition(self, java_blocks: List[CodeBlock],
                              target_blocks: List[CodeBlock],
                              mappings: List[CodeMapping]) -> None:
        """Display code transition visualization using Streamlit components"""
        # Set target type from the first block's language and ensure lowercase
        self.target_type = target_blocks[0].language.lower() if target_blocks else "ptx"
        
        # Define title mapping for different target types
        title_map = {
            "ptx": "PTX Target",
            "opencl": "OpenCL Target"
        }
        target_title = title_map.get(self.target_type, f"{self.target_type.upper()} Target")
        
        # Combine all Java code and target code
        java_code = self._combine_blocks(java_blocks)
        target_code = self._combine_blocks(target_blocks)
        
        # Find patterns in both code bases
        self.java_patterns = self._find_java_patterns(java_code)
        if self.target_type == 'ptx':
            self.target_patterns = self._find_ptx_patterns(target_code)
        else:
            self.target_patterns = self._find_opencl_patterns(target_code)
        
        # Create semantic mappings
        semantic_mappings = self._create_semantic_mappings()
        
        # Create the visualization HTML with proper title
        html = self._create_visualization_html(java_code, target_code, target_title, semantic_mappings)
        
        # Display using Streamlit components
        components.html(html, height=800, scrolling=True)
        
        # Display mapping statistics
        self._display_mapping_stats()
    
    def _combine_blocks(self, blocks: List[CodeBlock]) -> str:
        """Combine code blocks into a single string"""
        return "\n".join(block.code for block in blocks)
    
    def _find_java_patterns(self, code: str) -> Dict[str, List[Dict]]:
        """Find patterns in Java code"""
        patterns = {
            'array_access': [],
            'math_operations': [],
            'parallel_loops': [],
            'sequential_loops': [],
            'method_signatures': []
        }
        
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            # Array access patterns
            if '.get(' in line or '.set(' in line:
                patterns['array_access'].append({
                    'line': i,
                    'code': line.strip(),
                    'type': 'get' if '.get(' in line else 'set'
                })
            
            # Math operations
            if 'TornadoMath.' in line:
                match = re.search(r'TornadoMath\.(\w+)\(', line)
                if match:
                    patterns['math_operations'].append({
                        'line': i,
                        'code': line.strip(),
                        'operation': match.group(1)
                    })
            
            # Parallel loops
            if '@Parallel' in line:
                patterns['parallel_loops'].append({
                    'line': i,
                    'code': line.strip()
                })
            
            # Sequential loops
            elif 'for(' in line or 'while(' in line:
                patterns['sequential_loops'].append({
                    'line': i,
                    'code': line.strip()
                })
            
            # Method signatures
            if re.search(r'(public|private|protected)\s+\w+\s+\w+\s*\(', line):
                patterns['method_signatures'].append({
                    'line': i,
                    'code': line.strip()
                })
        
        return patterns
    
    def _find_ptx_patterns(self, code: str) -> Dict[str, List[Dict]]:
        """Find detailed patterns in PTX code with enhanced DFT-specific detection"""
        patterns = {
            'global_memory': [],
            'shared_memory': [],
            'registers': [],
            'math_operations': [],
            'control_flow': [],
            'thread_operations': [],
            'dft_specific': []  # New category for DFT-specific patterns
        }
        
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            # Global memory operations with DFT context
            if 'ld.global' in line:
                patterns['global_memory'].append({
                    'line': i,
                    'code': line.strip(),
                    'type': 'load',
                    'context': 'dft_input' if 'rud1' in line or 'rud2' in line else 'dft_output'
                })
            elif 'st.global' in line:
                patterns['global_memory'].append({
                    'line': i,
                    'code': line.strip(),
                    'type': 'store',
                    'context': 'dft_output'
                })
            
            # Register declarations and operations with type tracking
            if '.reg' in line:
                reg_type = re.search(r'\.reg\s+\.(\w+)', line)
                if reg_type:
                    patterns['registers'].append({
                        'line': i,
                        'code': line.strip(),
                        'type': 'declaration',
                        'reg_type': reg_type.group(1)
                    })
            elif re.search(r'mov\.\w+', line):
                patterns['registers'].append({
                    'line': i,
                    'code': line.strip(),
                    'type': 'move',
                    'reg_type': re.search(r'mov\.(\w+)', line).group(1)
                })
            
            # Math operations with precision tracking
            if any(op in line for op in ['sin.approx', 'cos.approx', 'mul.rn', 'add.rn', 'div.full', 'mad.rn']):
                op_match = re.search(r'(sin\.approx|cos\.approx|mul\.rn|add\.rn|div\.full|mad\.rn)', line)
                if op_match:
                    patterns['math_operations'].append({
                        'line': i,
                        'code': line.strip(),
                        'operation': op_match.group(1),
                        'precision': 'approx' if 'approx' in line else 'rn' if 'rn' in line else 'full'
                    })
            
            # Control flow with loop detection
            if any(term in line for term in ['bra', 'setp', 'ret']):
                ctrl_type = 'branch' if 'bra' in line else 'predicate' if 'setp' in line else 'return'
                patterns['control_flow'].append({
                    'line': i,
                    'code': line.strip(),
                    'type': ctrl_type,
                    'is_loop': 'LOOP_COND' in line
                })
            
            # Thread operations with dimension tracking
            if any(term in line for term in ['%tid', '%ntid', '%ctaid']):
                dim_match = re.search(r'%(\w+)\.(\w+)', line)
                if dim_match:
                    patterns['thread_operations'].append({
                        'line': i,
                        'code': line.strip(),
                        'type': dim_match.group(1),
                        'dimension': dim_match.group(2)
                    })
            
            # DFT-specific patterns
            if any(term in line for term in ['mul.wide.u32', 'cvt.s32.u64', 'mad.lo.s32']):
                patterns['dft_specific'].append({
                    'line': i,
                    'code': line.strip(),
                    'type': 'thread_indexing',
                    'operation': re.search(r'(\w+\.\w+)', line).group(1)
                })
            elif any(term in line for term in ['cvt.rn.f32.s32', 'mul.rn.f32', 'div.full.f32']):
                patterns['dft_specific'].append({
                    'line': i,
                    'code': line.strip(),
                    'type': 'angle_calculation',
                    'operation': re.search(r'(\w+\.\w+)', line).group(1)
                })
        
        return patterns
    
    def _find_opencl_patterns(self, code: str) -> Dict[str, List[Dict]]:
        """Find patterns in OpenCL code with enhanced GPU-specific features"""
        patterns = {
            'global_memory': [],
            'local_memory': [],
            'private_memory': [],
            'math_operations': [],
            'thread_setup': [],
            'loop_structures': [],
            'barriers': [],
            'atomic_operations': [],
            'vector_operations': []
        }
        
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            # Global memory operations
            if '__global' in line:
                is_store = '=' in line and line.index('=') > line.index('__global')
                patterns['global_memory'].append({
                    'line': i,
                    'code': line.strip(),
                    'type': 'store' if is_store else 'load',
                    'vector_width': self._detect_vector_width(line)
                })
            
            # Local memory operations
            if '__local' in line:
                is_store = '=' in line and line.index('=') > line.index('__local')
                patterns['local_memory'].append({
                    'line': i,
                    'code': line.strip(),
                    'type': 'store' if is_store else 'load'
                })
            
            # Private memory operations
            if '__private' in line or 'auto' in line:
                patterns['private_memory'].append({
                    'line': i,
                    'code': line.strip(),
                    'type': 'declaration' if 'auto' in line else 'operation'
                })
            
            # Math operations with extended support
            math_ops = ['native_', 'half_', 'fast_', 'fma', 'mad']
            if any(op in line for op in math_ops):
                match = re.search(r'((?:native|half|fast)_\w+|fma|mad)', line)
                if match:
                    patterns['math_operations'].append({
                        'line': i,
                        'code': line.strip(),
                        'operation': match.group(1),
                        'precision': 'fast' if 'native_' in line else 'precise'
                    })
            
            # Thread/work-item setup
            if any(term in line for term in ['get_global_id', 'get_local_id', 'get_group_id']):
                match = re.search(r'get_(\w+)_id\s*\(\s*(\d+)\s*\)', line)
                if match:
                    patterns['thread_setup'].append({
                        'line': i,
                        'code': line.strip(),
                        'scope': match.group(1),
                        'dimension': match.group(2)
                    })
            
            # Loop structures with stride detection
            if 'for(' in line or 'while(' in line:
                stride_match = re.search(r'(\+\+|--|\+=\s*\d+)', line)
                patterns['loop_structures'].append({
                    'line': i,
                    'code': line.strip(),
                    'type': 'for' if 'for(' in line else 'while',
                    'stride': stride_match.group(1) if stride_match else None
                })
            
            # Barrier operations
            if 'barrier(' in line:
                match = re.search(r'barrier\s*\(\s*(CLK_[^)]+)\s*\)', line)
                if match:
                    patterns['barriers'].append({
                        'line': i,
                        'code': line.strip(),
                        'scope': match.group(1)
                    })
            
            # Atomic operations
            if 'atomic_' in line:
                match = re.search(r'atomic_(\w+)', line)
                if match:
                    patterns['atomic_operations'].append({
                        'line': i,
                        'code': line.strip(),
                        'operation': match.group(1)
                    })
            
            # Vector operations
            vector_types = ['float2', 'float4', 'float8', 'float16', 
                          'int2', 'int4', 'int8', 'int16']
            if any(vtype in line for vtype in vector_types):
                patterns['vector_operations'].append({
                    'line': i,
                    'code': line.strip(),
                    'vector_type': next(vt for vt in vector_types if vt in line)
                })
        
        return patterns
    
    def _detect_vector_width(self, line: str) -> int:
        """Detect vector width from OpenCL code line"""
        vector_types = {
            'float2': 2, 'float4': 4, 'float8': 8, 'float16': 16,
            'int2': 2, 'int4': 4, 'int8': 8, 'int16': 16
        }
        for vtype, width in vector_types.items():
            if vtype in line:
                return width
        return 1
    
    def _create_semantic_mappings(self) -> List[Dict]:
        """Create detailed semantic mappings with DFT-specific patterns"""
        mappings = []
        
        # Map array access to memory operations with DFT context
        for java_access in self.java_patterns.get('array_access', []):
            target_pattern_name = 'global_memory'
            target_access_list = self.target_patterns.get(target_pattern_name, [])
            
            # Map .get() to load operations with DFT context
            if java_access['type'] == 'get':
                for target_access in target_access_list:
                    if target_access['type'] == 'load':
                        context = target_access.get('context', '')
                        mappings.append({
                            'source_lines': [java_access['line']],
                            'target_lines': [target_access['line']],
                            'type': 'array_access',
                            'subtype': f"get_to_load_{context}",
                            'description': f"Java array get → {self.target_type.upper()} memory load ({context})",
                            'confidence': 0.85
                        })
            
            # Map .set() to store operations with DFT context
            elif java_access['type'] == 'set':
                for target_access in target_access_list:
                    if target_access['type'] == 'store':
                        context = target_access.get('context', '')
                        mappings.append({
                            'source_lines': [java_access['line']],
                            'target_lines': [target_access['line']],
                            'type': 'array_access',
                            'subtype': f"set_to_store_{context}",
                            'description': f"Java array set → {self.target_type.upper()} memory store ({context})",
                            'confidence': 0.85
                        })
        
        # Map math operations with precision tracking
        for java_math in self.java_patterns.get('math_operations', []):
            target_math_list = self.target_patterns.get('math_operations', [])
            
            java_op = java_math['operation'].lower()
            for target_math in target_math_list:
                target_op = target_math.get('operation', '').lower()
                precision = target_math.get('precision', '')
                
                # Match similar math operations with precision
                if (java_op in target_op) or (java_op == 'sin' and 'sin' in target_op) or (java_op == 'cos' and 'cos' in target_op):
                    mappings.append({
                        'source_lines': [java_math['line']],
                        'target_lines': [target_math['line']],
                        'type': 'math_operation',
                        'subtype': f"{java_op}_to_{target_op.split('.')[0]}_{precision}",
                        'description': f"TornadoMath.{java_op} → {self.target_type.upper()} {target_op} ({precision})",
                        'confidence': 0.9
                    })
        
        # Map DFT-specific patterns
        dft_patterns = self.target_patterns.get('dft_specific', [])
        for dft_pattern in dft_patterns:
            if dft_pattern['type'] == 'thread_indexing':
                # Map to parallel loop setup
                for java_loop in self.java_patterns.get('parallel_loops', []):
                    mappings.append({
                        'source_lines': [java_loop['line']],
                        'target_lines': [dft_pattern['line']],
                        'type': 'dft_threading',
                        'subtype': 'parallel_to_thread_index',
                        'description': f"@Parallel loop → {self.target_type.upper()} thread indexing",
                        'confidence': 0.8
                    })
            elif dft_pattern['type'] == 'angle_calculation':
                # Map to math operations in Java
                for java_math in self.java_patterns.get('math_operations', []):
                    if 'angle' in java_math.get('code', '').lower():
                        mappings.append({
                            'source_lines': [java_math['line']],
                            'target_lines': [dft_pattern['line']],
                            'type': 'dft_math',
                            'subtype': 'angle_calculation',
                            'description': f"DFT angle calculation → {self.target_type.upper()} math operations",
                            'confidence': 0.85
                        })
        
        # Map control flow with loop detection
        for java_loop in self.java_patterns.get('sequential_loops', []):
            target_ctrl_list = self.target_patterns.get('control_flow', [])
            
            for target_ctrl in target_ctrl_list:
                if target_ctrl.get('is_loop'):
                    mappings.append({
                        'source_lines': [java_loop['line']],
                        'target_lines': [target_ctrl['line']],
                        'type': 'loop_structure',
                        'subtype': 'sequential_to_ptx_loop',
                        'description': f"Sequential loop → {self.target_type.upper()} loop structure",
                        'confidence': 0.7
                    })
        
        return mappings
    
    def _create_visualization_html(self, java_code: str, target_code: str, 
                                 target_title: str, mappings: List[Dict]) -> str:
        """Create HTML for the visualization with enhanced styling"""
        viz_id = f"viz_{hash(java_code + target_code)}"
        
        # Add syntax highlighting for Java code
        java_highlighted = self._highlight_java_code(java_code)
        target_highlighted = self._highlight_target_code(target_code, self.target_type)
        
        html = f"""
        <div id="{viz_id}" class="code-viz">
            <style>
                .code-viz {{
                    display: flex;
                    flex-direction: column;
                    height: 100%;
                    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
                    font-size: 14px;
                    line-height: 1.6;
                    background: #1e1e1e;
                    color: #d4d4d4;
                }}
                .code-panels {{
                    display: flex;
                    gap: 1.5rem;
                    flex: 1;
                    min-height: 400px;
                    height: calc(100vh - 300px);
                    max-height: 800px;
                    padding: 1rem;
                    overflow: hidden;
                }}
                .code-panel {{
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    border: 1px solid #3c3c3c;
                    border-radius: 8px;
                    background: #252526;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    min-width: 0;
                }}
                .panel-header {{
                    padding: 1rem;
                    background: #323233;
                    border-bottom: 1px solid #3c3c3c;
                    border-top-left-radius: 8px;
                    border-top-right-radius: 8px;
                    font-size: 16px;
                    font-weight: 600;
                    color: #ffffff;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    position: sticky;
                    top: 0;
                    z-index: 10;
                }}
                .code-content {{
                    flex: 1;
                    overflow: auto;
                    padding: 1rem;
                    font-size: 14px;
                    line-height: 1.6;
                    color: #e0e0e0;
                }}
                .line {{
                    display: flex;
                    padding: 2px 8px;
                    border-radius: 4px;
                    margin: 1px 0;
                    transition: all 0.2s ease;
                    white-space: pre;
                    width: 100%;
                }}
                .line:hover {{
                    background: rgba(86, 156, 214, 0.15);
                }}
                .line-number {{
                    color: #858585;
                    margin-right: 1.5rem;
                    user-select: none;
                    min-width: 3em;
                    text-align: right;
                }}
                .line-content {{
                    flex: 1;
                    white-space: pre;
                    font-family: inherit;
                    overflow-x: auto;
                }}
                .line.highlighted {{
                    background: rgba(86, 156, 214, 0.4);
                    border-left: 3px solid #569cd6;
                    font-weight: 500;
                    color: #ffffff;
                }}
                .line.highlighted-array {{
                    background: rgba(206, 145, 120, 0.4);
                    border-left: 3px solid #ce9178;
                    color: #ffffff;
                }}
                .line.highlighted-math {{
                    background: rgba(197, 134, 192, 0.4);
                    border-left: 3px solid #c586c0;
                    color: #ffffff;
                }}
                .line.highlighted-memory {{
                    background: rgba(181, 206, 168, 0.4);
                    border-left: 3px solid #b5cea8;
                    color: #ffffff;
                }}
                .line.highlighted-thread {{
                    background: rgba(220, 220, 170, 0.4);
                    border-left: 3px solid #dcdcaa;
                    color: #ffffff;
                }}
                .mapping-info {{
                    margin: 1rem;
                    padding: 1.5rem;
                    background: #323233;
                    border-radius: 8px;
                    color: #e0e0e0;
                    border: 1px solid #3c3c3c;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    max-height: 300px;
                    overflow-y: auto;
                    resize: vertical;
                }}
                .mapping-info h3 {{
                    margin-top: 0;
                    font-size: 18px;
                    color: #ffffff;
                    font-weight: 600;
                    margin-bottom: 1rem;
                    position: sticky;
                    top: 0;
                    background: #323233;
                    padding: 0.5rem 0;
                    z-index: 1;
                }}
                .mapping-info div {{
                    margin: 8px 0;
                    padding: 12px;
                    background: #252526;
                    border-radius: 6px;
                    border-left: 3px solid #569cd6;
                    transition: transform 0.2s;
                }}
                .mapping-info div:hover {{
                    transform: translateX(4px);
                    background: #2d2d2d;
                }}
                .mapping-info strong {{
                    color: #569cd6;
                    font-weight: 600;
                }}
                .mapping-info hr {{
                    border: none;
                    border-top: 1px solid #3c3c3c;
                    margin: 12px 0;
                }}
                /* Enhanced syntax highlighting */
                .keyword {{ color: #569cd6; font-weight: 600; }}
                .string {{ color: #ce9178; }}
                .comment {{ color: #6a9955; font-style: italic; }}
                .type {{ color: #4ec9b0; font-weight: 500; }}
                .number {{ color: #b5cea8; }}
                .function {{ color: #dcdcaa; }}
                .operator {{ color: #d4d4d4; }}
                .variable {{ color: #9cdcfe; }}
                .annotation {{ color: #c586c0; font-style: italic; }}
                
                /* Scrollbar styling */
                .code-content::-webkit-scrollbar,
                .mapping-info::-webkit-scrollbar {{
                    width: 12px;
                    height: 12px;
                }}
                .code-content::-webkit-scrollbar-track,
                .mapping-info::-webkit-scrollbar-track {{
                    background: #1e1e1e;
                    border-radius: 6px;
                }}
                .code-content::-webkit-scrollbar-thumb,
                .mapping-info::-webkit-scrollbar-thumb {{
                    background: #424242;
                    border-radius: 6px;
                    border: 3px solid #1e1e1e;
                }}
                .code-content::-webkit-scrollbar-thumb:hover,
                .mapping-info::-webkit-scrollbar-thumb:hover {{
                    background: #4f4f4f;
                }}
                
                /* Responsive adjustments */
                @media (max-width: 1200px) {{
                    .code-panels {{
                        flex-direction: column;
                        height: auto;
                    }}
                    .code-panel {{
                        height: 400px;
                    }}
                }}
                @media (max-width: 768px) {{
                    .code-panel {{
                        height: 300px;
                    }}
                    .mapping-info {{
                        max-height: 200px;
                    }}
                }}
            </style>
            
            <div class="code-panels">
                <div class="code-panel">
                    <div class="panel-header">Java Source</div>
                    <div class="code-content" id="{viz_id}_java">
                        {java_highlighted}
                    </div>
                </div>
                
                <div class="code-panel">
                    <div class="panel-header">{target_title}</div>
                    <div class="code-content" id="{viz_id}_target">
                        {target_highlighted}
                    </div>
                </div>
            </div>
            
            <div class="mapping-info">
                <h3>Mapping Information</h3>
                <div id="{viz_id}_info"></div>
            </div>
            
            <script>
                (function() {{
                    const mappings = {json.dumps(mappings)};
                    const javaPanel = document.getElementById('{viz_id}_java');
                    const targetPanel = document.getElementById('{viz_id}_target');
                    const infoPanel = document.getElementById('{viz_id}_info');
                    
                    // Add event listeners for Java lines
                    javaPanel.querySelectorAll('.line').forEach(line => {{
                        line.addEventListener('mouseover', () => {{
                            const lineNum = parseInt(line.getAttribute('data-line'));
                            const relatedMappings = mappings.filter(m => m.source_lines.includes(lineNum));
                            
                            if (relatedMappings.length > 0) {{
                                // Highlight this line
                                line.classList.add('highlighted');
                                
                                // Highlight target lines
                                relatedMappings.forEach(mapping => {{
                                    mapping.target_lines.forEach(targetLineNum => {{
                                        const targetLine = targetPanel.querySelector(`[data-line="${{targetLineNum}}"]`);
                                        if (targetLine) {{
                                            targetLine.classList.add(`highlighted-${{mapping.type.toLowerCase()}}`);
                                            targetLine.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                                        }}
                                    }});
                                }});
                                
                                // Show mapping info
                                infoPanel.innerHTML = relatedMappings.map(m => `
                                    <div class="map-${{m.type.toLowerCase()}}">
                                        <strong>${{m.type}}</strong>: ${{m.description}}<br>
                                        Java line ${{m.source_lines.join(', ')}} → {self.target_type.upper()} line ${{m.target_lines.join(', ')}}
                                    </div>
                                `).join('<hr>');
                            }}
                        }});
                        
                        line.addEventListener('mouseout', () => {{
                            // Remove all highlights
                            document.querySelectorAll('.line').forEach(l => {{
                                l.classList.remove('highlighted');
                                l.classList.remove('highlighted-array');
                                l.classList.remove('highlighted-math');
                                l.classList.remove('highlighted-memory');
                                l.classList.remove('highlighted-thread');
                            }});
                            infoPanel.innerHTML = '';
                        }});
                    }});
                    
                    // Add event listeners for target lines
                    targetPanel.querySelectorAll('.line').forEach(line => {{
                        line.addEventListener('mouseover', () => {{
                            const lineNum = parseInt(line.getAttribute('data-line'));
                            const relatedMappings = mappings.filter(m => m.target_lines.includes(lineNum));
                            
                            if (relatedMappings.length > 0) {{
                                // Highlight this line
                                line.classList.add('highlighted');
                                
                                // Highlight Java lines
                                relatedMappings.forEach(mapping => {{
                                    mapping.source_lines.forEach(sourceLineNum => {{
                                        const javaLine = javaPanel.querySelector(`[data-line="${{sourceLineNum}}"]`);
                                        if (javaLine) {{
                                            javaLine.classList.add(`highlighted-${{mapping.type.toLowerCase()}}`);
                                            javaLine.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                                        }}
                                    }});
                                }});
                                
                                // Show mapping info
                                infoPanel.innerHTML = relatedMappings.map(m => `
                                    <div class="map-${{m.type.toLowerCase()}}">
                                        <strong>${{m.type}}</strong>: ${{m.description}}<br>
                                        {self.target_type.upper()} line ${{m.target_lines.join(', ')}} → Java line ${{m.source_lines.join(', ')}}
                                    </div>
                                `).join('<hr>');
                            }}
                        }});
                        
                        line.addEventListener('mouseout', () => {{
                            // Remove all highlights
                            document.querySelectorAll('.line').forEach(l => {{
                                l.classList.remove('highlighted');
                                l.classList.remove('highlighted-array');
                                l.classList.remove('highlighted-math');
                                l.classList.remove('highlighted-memory');
                                l.classList.remove('highlighted-thread');
                            }});
                            infoPanel.innerHTML = '';
                        }});
                    }});
                    
                    // Synchronize scrolling between panels
                    let isScrolling = false;
                    javaPanel.addEventListener('scroll', () => {{
                        if (!isScrolling) {{
                            isScrolling = true;
                            const scrollPercentage = javaPanel.scrollTop / (javaPanel.scrollHeight - javaPanel.clientHeight);
                            targetPanel.scrollTop = scrollPercentage * (targetPanel.scrollHeight - targetPanel.clientHeight);
                            setTimeout(() => isScrolling = false, 50);
                        }}
                    }});
                    
                    targetPanel.addEventListener('scroll', () => {{
                        if (!isScrolling) {{
                            isScrolling = true;
                            const scrollPercentage = targetPanel.scrollTop / (targetPanel.scrollHeight - targetPanel.clientHeight);
                            javaPanel.scrollTop = scrollPercentage * (javaPanel.scrollHeight - javaPanel.clientHeight);
                            setTimeout(() => isScrolling = false, 50);
                        }}
                    }});
                }})();
            </script>
        </div>
        """
        
        return html
    
    def _highlight_java_code(self, code: str) -> str:
        """Enhanced Java syntax highlighting with proper whitespace preservation"""
        # Define Java language elements
        java_keywords = [
            'public', 'private', 'protected', 'static', 'final', 'void',
            'int', 'float', 'double', 'boolean', 'for', 'while', 'do',
            'if', 'else', 'return', 'new', 'class', 'interface', 'extends',
            'implements', 'try', 'catch', 'finally', 'throw', 'throws',
            'synchronized', 'volatile', 'transient', 'native', 'package',
            'import', 'instanceof', 'super', 'this', 'abstract', 'continue',
            'break', 'assert', 'default', 'switch', 'case', 'enum'
        ]
        java_types = [
            'String', 'Integer', 'Float', 'Double', 'Boolean', 'List', 'Map',
            'FloatArray', 'Object', 'Class', 'Exception', 'RuntimeException',
            'Throwable', 'System', 'Thread', 'Runnable', 'Collection', 'Set',
            'Vector', 'ArrayList', 'HashMap', 'TreeMap', 'LinkedList', 'Queue',
            'Deque', 'Stack', 'StringBuilder', 'StringBuffer', 'Character',
            'Byte', 'Short', 'Long', 'Number', 'Math', 'TornadoMath'
        ]
        
        def escape_html(text: str) -> str:
            """Escape HTML special characters while preserving whitespace"""
            # First escape HTML special characters
            escaped = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            # Then convert leading spaces to non-breaking spaces
            leading_spaces = len(text) - len(text.lstrip())
            return '&nbsp;' * leading_spaces + escaped.lstrip()
        
        lines = code.split('\n')
        highlighted_lines = []
        
        for i, line in enumerate(lines, 1):
            # Convert the line to a list of tokens with their types
            tokens = []
            remaining_line = line
            
            # Handle comments first
            if '//' in remaining_line:
                code_part, comment = remaining_line.split('//', 1)
                remaining_line = code_part
                tokens.append(('comment', '//' + comment))
            elif '/*' in remaining_line and '*/' in remaining_line:
                before, rest = remaining_line.split('/*', 1)
                comment, after = rest.split('*/', 1)
                tokens.append(('text', before))
                tokens.append(('comment', '/*' + comment + '*/'))
                remaining_line = after
            
            # Process the remaining line
            while remaining_line:
                # Try to match each pattern in order of priority
                matched = False
                
                # 1. String literals with escape handling
                string_match = re.match(r'^"(?:[^"\\]|\\.)*"', remaining_line)
                if string_match:
                    tokens.append(('string', string_match.group()))
                    remaining_line = remaining_line[len(string_match.group()):]
                    matched = True
                    continue
                
                # 2. Character literals
                char_match = re.match(r"^'(?:[^'\\]|\\.)'", remaining_line)
                if char_match:
                    tokens.append(('string', char_match.group()))
                    remaining_line = remaining_line[len(char_match.group()):]
                    matched = True
                    continue
                
                # 3. Numbers (including float literals and hex)
                number_match = re.match(r'^\b(?:0x[0-9a-fA-F]+|[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?[fFdDlL]?)\b', remaining_line)
                if number_match:
                    tokens.append(('number', number_match.group()))
                    remaining_line = remaining_line[len(number_match.group()):]
                    matched = True
                    continue
                
                # 4. Annotations
                annotation_match = re.match(r'^@\w+', remaining_line)
                if annotation_match:
                    tokens.append(('annotation', annotation_match.group()))
                    remaining_line = remaining_line[len(annotation_match.group()):]
                    matched = True
                    continue
                
                # 5. Types (check before keywords as some types might contain keywords)
                for type_name in sorted(java_types, key=len, reverse=True):
                    if remaining_line.startswith(type_name) and (
                        len(remaining_line) == len(type_name) or 
                        not remaining_line[len(type_name)].isalnum() and remaining_line[len(type_name)] != '_'
                    ):
                        tokens.append(('type', type_name))
                        remaining_line = remaining_line[len(type_name):]
                        matched = True
                        break
                if matched:
                    continue
                
                # 6. Keywords
                for keyword in sorted(java_keywords, key=len, reverse=True):
                    if remaining_line.startswith(keyword) and (
                        len(remaining_line) == len(keyword) or 
                        not remaining_line[len(keyword)].isalnum() and remaining_line[len(keyword)] != '_'
                    ):
                        tokens.append(('keyword', keyword))
                        remaining_line = remaining_line[len(keyword):]
                        matched = True
                        break
                if matched:
                    continue
                
                # 7. Method calls
                method_match = re.match(r'^(\w+)\s*\(', remaining_line)
                if method_match:
                    tokens.append(('function', method_match.group(1)))
                    remaining_line = remaining_line[len(method_match.group(1)):]
                    matched = True
                    continue
                
                # 8. Operators
                operator_match = re.match(r'^([+\-*/=<>!&|^~%]=?|>=|<=|==|!=|&&|\|\||<<|>>|>>>|\+\+|--|\?|:|\.|,|;|\[|\]|\(|\)|\{|\})', remaining_line)
                if operator_match:
                    tokens.append(('operator', operator_match.group()))
                    remaining_line = remaining_line[len(operator_match.group()):]
                    matched = True
                    continue
                
                # 9. Variables and other identifiers
                identifier_match = re.match(r'^[a-zA-Z_]\w*', remaining_line)
                if identifier_match:
                    tokens.append(('variable', identifier_match.group()))
                    remaining_line = remaining_line[len(identifier_match.group()):]
                    matched = True
                    continue
                
                # If no match, take one character as plain text
                tokens.append(('text', remaining_line[0]))
                remaining_line = remaining_line[1:]
            
            # Build the highlighted line with preserved whitespace
            highlighted_code = []
            for token_type, token_text in tokens:
                escaped_text = escape_html(token_text)
                if token_type == 'text':
                    highlighted_code.append(escaped_text)
                else:
                    highlighted_code.append(f'<span class="{token_type}">{escaped_text}</span>')
            
            # Create the line div with proper classes and preserved whitespace
            highlighted_lines.append(
                f'<div class="line" data-line="{i}">'
                f'<span class="line-number">{i}</span>'
                f'<span class="line-content">{" ".join(highlighted_code)}</span>'
                f'</div>'
            )
        
        return '\n'.join(highlighted_lines)
    
    def _highlight_target_code(self, code: str, target_type: str) -> str:
        """Enhanced target code syntax highlighting"""
        if target_type.lower() == 'ptx':
            keywords = ['ld.global', 'st.global', 'mov', 'add', 'mul', 'div', 'bra', 'setp']
            types = ['%rd', '%r', '%f', '%p', '%b']
            special_funcs = []
        else:  # OpenCL
            keywords = [
                '__global', '__local', '__private', '__constant', 'kernel', 'void',
                'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break',
                'return', 'typedef', 'struct', 'union', 'volatile', 'const'
            ]
            types = [
                'float', 'double', 'int', 'long', 'char', 'unsigned', 'size_t', 'uint',
                'float2', 'float3', 'float4', 'float8', 'float16',
                'int2', 'int3', 'int4', 'int8', 'int16',
                'uint2', 'uint3', 'uint4', 'uint8', 'uint16'
            ]
            special_funcs = [
                'get_global_id', 'get_local_id', 'get_group_id',
                'get_global_size', 'get_local_size', 'get_num_groups',
                'barrier', 'mem_fence', 'atomic_add', 'atomic_sub',
                'native_sin', 'native_cos', 'native_exp', 'native_log',
                'mad', 'fma', 'clamp', 'min', 'max'
            ]

        lines = code.split('\n')
        highlighted_lines = []

        for i, line in enumerate(lines, 1):
            # Preserve leading whitespace
            leading_space = len(line) - len(line.lstrip())
            line_content = line.lstrip()
            
            # Handle comments
            if '//' in line_content:
                code_part, comment_part = line_content.split('//', 1)
                comment = f'<span class="comment">//{comment_part}</span>'
            else:
                code_part, comment = line_content, ''

            # Highlight special functions first (for OpenCL)
            for func in special_funcs:
                pattern = f'\\b{func}\\b'
                code_part = re.sub(pattern, f'<span class="function">{func}</span>', code_part)

            # Highlight types (before keywords to avoid partial matches)
            for type_name in sorted(types, key=len, reverse=True):
                pattern = f'\\b{type_name}\\b'
                code_part = re.sub(pattern, f'<span class="type">{type_name}</span>', code_part)

            # Highlight keywords
            for keyword in sorted(keywords, key=len, reverse=True):
                pattern = f'\\b{keyword}\\b'
                code_part = re.sub(pattern, f'<span class="keyword">{keyword}</span>', code_part)

            # Highlight numbers (including floating point)
            code_part = re.sub(
                r'\b\d+\.?\d*(?:[eE][-+]?\d+)?[fFdDuUlL]*\b',
                lambda m: f'<span class="number">{m.group()}</span>',
                code_part
            )

            # Highlight operators
            code_part = re.sub(
                r'([-+*/=<>!&|^~%]=?|>=|<=|==|!=|&&|\|\||<<|>>|\+\+|--)',
                r'<span class="operator">\1</span>',
                code_part
            )

            # Add back leading whitespace using non-breaking spaces
            leading_whitespace = '&nbsp;' * leading_space

            highlighted_lines.append(
                f'<div class="line" data-line="{i}">'
                f'<span class="line-number">{i}</span>'
                f'<span class="line-content">{leading_whitespace}{code_part}{comment}</span>'
                f'</div>'
            )

        return '\n'.join(highlighted_lines)
    
    def _display_mapping_stats(self) -> None:
        """Display enhanced mapping statistics with DFT-specific metrics"""
        if not self.mapping_stats:
            st.warning("No mapping statistics available.")
            return
        
        st.header("Code Mapping Statistics")
        
        # Summary metrics in columns
        cols = st.columns(3)
        with cols[0]:
            st.metric("Total Mappings", self.mapping_stats['total_mappings'])
        
        with cols[1]:
            high_conf = self.mapping_stats['confidence']['high']
            high_pct = (high_conf / self.mapping_stats['total_mappings'] * 100) if self.mapping_stats['total_mappings'] > 0 else 0
            st.metric("High Confidence Mappings", f"{high_conf} ({high_pct:.1f}%)")
        
        with cols[2]:
            if self.target_type == 'ptx':
                st.metric("DFT-Specific Patterns", len(self.target_patterns.get('dft_specific', [])))
        
        # Display mappings by type with DFT context
        st.subheader("Mappings by Type")
        
        # Create DataFrame for bar chart
        mapping_types = list(self.mapping_stats['by_type'].keys())
        mapping_counts = list(self.mapping_stats['by_type'].values())
        
        df = pd.DataFrame({
            'Type': [t.replace('_', ' ').title() for t in mapping_types],
            'Count': mapping_counts
        })
        
        # Create bar chart with DFT-specific highlighting
        if not df.empty:
            fig = px.bar(df, x='Type', y='Count', 
                      title='Distribution of Mapping Types',
                      color='Type',
                      color_discrete_sequence=px.colors.qualitative.G10)
            
            fig.update_layout(
                xaxis_title="Mapping Type",
                yaxis_title="Number of Mappings",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # DFT-specific metrics
        if self.target_type == 'ptx':
            st.subheader("DFT-Specific Analysis")
            
            # Create metric columns for DFT patterns
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("Thread Indexing", len([p for p in self.target_patterns.get('dft_specific', []) 
                                                if p['type'] == 'thread_indexing']))
            with metric_cols[1]:
                st.metric("Angle Calculations", len([p for p in self.target_patterns.get('dft_specific', []) 
                                                   if p['type'] == 'angle_calculation']))
            with metric_cols[2]:
                st.metric("Global Memory Ops", len(self.target_patterns.get('global_memory', [])))
            with metric_cols[3]:
                st.metric("Math Operations", len(self.target_patterns.get('math_operations', [])))
            
            # Create pie chart for DFT memory operations
            memory_df = pd.DataFrame({
                'Type': ['Input Loads', 'Output Stores', 'Math Operations'],
                'Count': [
                    len([p for p in self.target_patterns.get('global_memory', []) 
                         if p['type'] == 'load' and p.get('context') == 'dft_input']),
                    len([p for p in self.target_patterns.get('global_memory', []) 
                         if p['type'] == 'store' and p.get('context') == 'dft_output']),
                    len(self.target_patterns.get('math_operations', []))
                ]
            })
            
            fig = px.pie(memory_df, values='Count', names='Type',
                      title='DFT Memory and Math Operations',
                      color_discrete_sequence=px.colors.sequential.Plasma)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Display detailed mapping subtypes
        st.subheader("Detailed Mapping Subtypes")
        
        # Create DataFrame for detailed subtypes
        subtypes = list(self.mapping_stats['by_subtype'].keys())
        subtype_counts = list(self.mapping_stats['by_subtype'].values())
        
        detailed_df = pd.DataFrame({
            'Subtype': [s.replace('_', ' ').replace('to', '→').title() for s in subtypes],
            'Count': subtype_counts
        })
        
        # Create horizontal bar chart
        if not detailed_df.empty:
            detailed_df = detailed_df.sort_values('Count', ascending=True)
            
            fig = px.bar(detailed_df, y='Subtype', x='Count', 
                      title='Detailed Mapping Subtypes',
                      orientation='h',
                      color='Count',
                      color_continuous_scale=px.colors.sequential.Plasma)
            
            fig.update_layout(
                yaxis_title="",
                xaxis_title="Number of Mappings",
                showlegend=False,
                height=max(300, len(subtypes) * 30)
            )
            
            st.plotly_chart(fig, use_container_width=True)

    def display_performance_analysis(self, blocks: List[CodeBlock]) -> None:
        """Display performance analysis using Streamlit components"""
        # Group blocks by language
        blocks_by_language = {}
        for block in blocks:
            if block.language not in blocks_by_language:
                blocks_by_language[block.language] = []
            blocks_by_language[block.language].append(block)
        
        # Create tabs for each language
        if not blocks_by_language:
            st.warning("No blocks with performance metrics found.")
            return
        
        # Create language tabs
        tabs = st.tabs(list(blocks_by_language.keys()))
        
        for lang, tab in zip(blocks_by_language.keys(), tabs):
            with tab:
                blocks = blocks_by_language[lang]
                
                # Create metrics overview
                st.subheader("Performance Metrics Overview")
                
                # Calculate aggregate metrics
                total_blocks = len(blocks)
                blocks_with_metrics = sum(1 for b in blocks if b.metrics)
                
                # Display basic stats
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Blocks", total_blocks)
                with col2:
                    st.metric("Blocks with Metrics", blocks_with_metrics)
                
                # Display detailed metrics for each block
                st.subheader("Block-level Metrics")
                
                for i, block in enumerate(blocks):
                    with st.expander(f"Block {i} (Lines {block.line_start}-{block.line_end})"):
                        # Show code
                        st.code(block.code, language=lang.lower())
                        
                        if block.metrics:
                            # Create columns for different metric categories
                            metric_cols = st.columns(3)
                            
                            # Instruction mix
                            with metric_cols[0]:
                                st.markdown("##### Instruction Mix")
                                if 'instruction_mix' in block.metrics:
                                    # Create pie chart
                                    data = block.metrics['instruction_mix']
                                    st.plotly_chart(self._create_pie_chart(
                                        data, 
                                        "Instruction Types",
                                        "Percentage"
                                    ), use_container_width=True)
                                else:
                                    st.info("No instruction mix data available")
                            
                            # Memory access
                            with metric_cols[1]:
                                st.markdown("##### Memory Access")
                                if 'memory_access' in block.metrics:
                                    data = block.metrics['memory_access']
                                    st.plotly_chart(self._create_bar_chart(
                                        data,
                                        "Access Type",
                                        "Count",
                                        "Memory Access Patterns"
                                    ), use_container_width=True)
                                else:
                                    st.info("No memory access data available")
                            
                            # Thread divergence
                            with metric_cols[2]:
                                st.markdown("##### Thread Divergence")
                                if 'thread_divergence' in block.metrics:
                                    divergence = block.metrics['thread_divergence']
                                    st.metric(
                                        "Divergence Score",
                                        f"{divergence:.2%}",
                                        delta="-" if divergence > 0.2 else "+"
                                    )
                                else:
                                    st.info("No thread divergence data available")
                        else:
                            st.warning("No performance metrics available for this block")
                
                # Show aggregate metrics
                st.subheader("Aggregate Metrics")
                
                # Combine metrics from all blocks
                combined_metrics = self._combine_block_metrics(blocks)
                
                if combined_metrics:
                    metric_cols = st.columns(3)
                    
                    with metric_cols[0]:
                        st.markdown("##### Overall Instruction Distribution")
                        if 'instruction_mix' in combined_metrics:
                            st.plotly_chart(self._create_pie_chart(
                                combined_metrics['instruction_mix'],
                                "Instruction Types",
                                "Percentage"
                            ), use_container_width=True)
                    
                    with metric_cols[1]:
                        st.markdown("##### Memory Access Summary")
                        if 'memory_access' in combined_metrics:
                            st.plotly_chart(self._create_bar_chart(
                                combined_metrics['memory_access'],
                                "Access Type",
                                "Count",
                                "Memory Access Patterns"
                            ), use_container_width=True)
                    
                    with metric_cols[2]:
                        st.markdown("##### Overall Thread Divergence")
                        if 'thread_divergence' in combined_metrics:
                            avg_divergence = combined_metrics['thread_divergence']
                            st.metric(
                                "Average Divergence",
                                f"{avg_divergence:.2%}",
                                delta="-" if avg_divergence > 0.2 else "+"
                            )
    
    def _create_pie_chart(self, data: Dict[str, float], title: str, value_label: str) -> go.Figure:
        """Create a pie chart using plotly"""
        fig = go.Figure(data=[go.Pie(
            labels=list(data.keys()),
            values=list(data.values()),
            hole=0.3,
            textinfo='label+percent'
        )])
        
        fig.update_layout(
            title=title,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=300
        )
        
        return fig
    
    def _create_bar_chart(self, data: Dict[str, int], x_label: str, y_label: str, title: str) -> go.Figure:
        """Create a bar chart using plotly"""
        fig = go.Figure(data=[go.Bar(
            x=list(data.keys()),
            y=list(data.values())
        )])
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            height=300
        )
        
        return fig
    
    def _combine_block_metrics(self, blocks: List[CodeBlock]) -> Dict:
        """Combine metrics from multiple blocks"""
        combined = {
            'instruction_mix': {},
            'memory_access': {},
            'thread_divergence': 0.0
        }
        
        blocks_with_metrics = [b for b in blocks if b.metrics]
        if not blocks_with_metrics:
            return None
        
        # Combine instruction mix
        for block in blocks_with_metrics:
            if 'instruction_mix' in block.metrics:
                for instr, count in block.metrics['instruction_mix'].items():
                    if instr not in combined['instruction_mix']:
                        combined['instruction_mix'][instr] = 0
                    combined['instruction_mix'][instr] += count
        
        # Combine memory access
        for block in blocks_with_metrics:
            if 'memory_access' in block.metrics:
                for access, count in block.metrics['memory_access'].items():
                    if access not in combined['memory_access']:
                        combined['memory_access'][access] = 0
                    combined['memory_access'][access] += count
        
        # Average thread divergence
        divergence_blocks = [b for b in blocks_with_metrics if 'thread_divergence' in b.metrics]
        if divergence_blocks:
            combined['thread_divergence'] = sum(
                b.metrics['thread_divergence'] for b in divergence_blocks
            ) / len(divergence_blocks)
        
        return combined

    def display_control_flow(self, blocks: List[CodeBlock], edges: List[Tuple[int, int]]) -> None:
        """Display control flow visualization using Streamlit components"""
        import graphviz
        import streamlit as st
        
        # Create a new Graphviz graph
        dot = graphviz.Digraph(comment='Control Flow Graph')
        dot.attr(rankdir='LR')  # Left to right layout
        
        # Add nodes (code blocks)
        for i, block in enumerate(blocks):
            # Create label with code preview
            label = f"{block.language.upper()} Block {i}\n"
            # Add first line of code as preview (truncated)
            preview = block.code.split('\n')[0][:30] + '...' if block.code else 'Empty block'
            label += f"{preview}\n"
            label += f"Lines {block.line_start}-{block.line_end}"
            
            # Add node with styling based on language
            color = {
                'java': '#4CAF50',    # Green for Java
                'ptx': '#2196F3',     # Blue for PTX
                'opencl': '#FF9800'   # Orange for OpenCL
            }.get(block.language.lower(), '#9E9E9E')
            
            dot.node(str(i), label, 
                    style='filled',
                    fillcolor=color,
                    fontcolor='white',
                    shape='box')
        
        # Add edges
        for src, dst in edges:
            dot.edge(str(src), str(dst))
        
        # Display the graph
        st.graphviz_chart(dot)
        
        # Add code viewer below the graph
        st.markdown("### Code Blocks")
        for i, block in enumerate(blocks):
            with st.expander(f"{block.language.upper()} Block {i} (Lines {block.line_start}-{block.line_end})"):
                st.code(block.code, language=block.language.lower()) 