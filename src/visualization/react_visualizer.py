import streamlit as st
import streamlit.components.v1 as components
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import re
import plotly.graph_objects as go

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
    
    def display_code_transition(self, java_blocks: List[CodeBlock],
                              target_blocks: List[CodeBlock],
                              mappings: List[CodeMapping]) -> None:
        """Display code transition visualization using Streamlit components"""
        # Combine all Java code and target code
        java_code = self._combine_blocks(java_blocks)
        target_code = self._combine_blocks(target_blocks)
        target_type = target_blocks[0].language if target_blocks else "ptx"
        
        # Find patterns in both code bases
        self.java_patterns = self._find_java_patterns(java_code)
        if target_type.lower() == 'ptx':
            self.target_patterns = self._find_ptx_patterns(target_code)
        else:
            self.target_patterns = self._find_opencl_patterns(target_code)
        
        # Create semantic mappings
        semantic_mappings = self._create_semantic_mappings()
        
        # Create the visualization HTML
        html = self._create_visualization_html(java_code, target_code, target_type, semantic_mappings)
        
        # Display using Streamlit components
        components.html(html, height=800, scrolling=True)
        
        # Display mapping statistics
        self._display_mapping_stats(semantic_mappings)
    
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
        """Find patterns in PTX code"""
        patterns = {
            'memory_operations': [],
            'math_operations': [],
            'thread_setup': [],
            'loop_structures': []
        }
        
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            # Memory operations
            if 'ld.global' in line or 'st.global' in line:
                patterns['memory_operations'].append({
                    'line': i,
                    'code': line.strip(),
                    'type': 'load' if 'ld.global' in line else 'store'
                })
            
            # Math operations
            if any(op in line for op in ['sin.approx', 'cos.approx', 'mul.rn', 'add.rn']):
                match = re.search(r'(sin\.approx|cos\.approx|mul\.rn|add\.rn)', line)
                if match:
                    patterns['math_operations'].append({
                        'line': i,
                        'code': line.strip(),
                        'operation': match.group(1)
                    })
            
            # Thread setup
            if any(term in line for term in ['%tid', '%ntid', 'mov.u32']):
                patterns['thread_setup'].append({
                    'line': i,
                    'code': line.strip()
                })
            
            # Loop structures
            if any(term in line for term in ['@!P', 'bra', 'setp']):
                patterns['loop_structures'].append({
                    'line': i,
                    'code': line.strip()
                })
        
        return patterns
    
    def _find_opencl_patterns(self, code: str) -> Dict[str, List[Dict]]:
        """Find patterns in OpenCL code"""
        patterns = {
            'memory_operations': [],
            'math_operations': [],
            'thread_setup': [],
            'loop_structures': []
        }
        
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            # Memory operations
            if '__global' in line:
                patterns['memory_operations'].append({
                    'line': i,
                    'code': line.strip(),
                    'type': 'store' if '=' in line else 'load'
                })
            
            # Math operations
            if any(op in line for op in ['native_sin', 'native_cos', 'native_sqrt']):
                match = re.search(r'(native_\w+)', line)
                if match:
                    patterns['math_operations'].append({
                        'line': i,
                        'code': line.strip(),
                        'operation': match.group(1)
                    })
            
            # Thread setup
            if 'get_global_id' in line or 'get_local_id' in line:
                patterns['thread_setup'].append({
                    'line': i,
                    'code': line.strip()
                })
            
            # Loop structures
            if 'for(' in line or 'while(' in line:
                patterns['loop_structures'].append({
                    'line': i,
                    'code': line.strip()
                })
        
        return patterns
    
    def _create_semantic_mappings(self) -> List[Dict]:
        """Create semantic mappings between Java and target code patterns"""
        mappings = []
        
        # Map array access to memory operations
        for java_access in self.java_patterns['array_access']:
            for target_mem in self.target_patterns['memory_operations']:
                if java_access['type'] == 'get' and target_mem['type'] == 'load':
                    mappings.append({
                        'source_line': java_access['line'],
                        'target_line': target_mem['line'],
                        'type': 'array_access',
                        'description': 'Array get operation maps to memory load'
                    })
                elif java_access['type'] == 'set' and target_mem['type'] == 'store':
                    mappings.append({
                        'source_line': java_access['line'],
                        'target_line': target_mem['line'],
                        'type': 'array_access',
                        'description': 'Array set operation maps to memory store'
                    })
        
        # Map math operations
        for java_math in self.java_patterns['math_operations']:
            for target_math in self.target_patterns['math_operations']:
                if java_math['operation'].lower() in target_math['operation'].lower():
                    mappings.append({
                        'source_line': java_math['line'],
                        'target_line': target_math['line'],
                        'type': 'math_operation',
                        'description': f"Math operation {java_math['operation']} maps to {target_math['operation']}"
                    })
        
        # Map parallel loops to thread setup
        for java_loop in self.java_patterns['parallel_loops']:
            for target_thread in self.target_patterns['thread_setup']:
                mappings.append({
                    'source_line': java_loop['line'],
                    'target_line': target_thread['line'],
                    'type': 'parallel_loop',
                    'description': 'Parallel loop maps to thread setup'
                })
        
        # Map sequential loops
        for java_loop in self.java_patterns['sequential_loops']:
            for target_loop in self.target_patterns['loop_structures']:
                mappings.append({
                    'source_line': java_loop['line'],
                    'target_line': target_loop['line'],
                    'type': 'sequential_loop',
                    'description': 'Sequential loop maps to loop structure'
                })
        
        return mappings
    
    def _create_visualization_html(self, java_code: str, target_code: str, 
                                 target_type: str, mappings: List[Dict]) -> str:
        """Create HTML for the visualization"""
        # Create unique ID for this visualization
        viz_id = f"viz_{hash(java_code + target_code)}"
        
        # Create the HTML template with the visualization
        html = f"""
        <div id="{viz_id}" class="code-viz">
            <style>
                .code-viz {{
                    display: flex;
                    flex-direction: column;
                    height: 100%;
                    font-family: monospace;
                }}
                .code-panels {{
                    display: flex;
                    gap: 1rem;
                    height: 600px;
                }}
                .code-panel {{
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                }}
                .panel-header {{
                    padding: 0.5rem;
                    background: #f0f0f0;
                    border-bottom: 1px solid #ccc;
                }}
                .code-content {{
                    flex: 1;
                    overflow: auto;
                    padding: 0.5rem;
                }}
                .line {{
                    display: flex;
                    padding: 2px 4px;
                }}
                .line-number {{
                    color: #666;
                    margin-right: 1rem;
                    user-select: none;
                }}
                .line.highlighted {{
                    background: #e6f3ff;
                }}
                .mapping-info {{
                    margin-top: 1rem;
                    padding: 1rem;
                    background: #f8f9fa;
                    border-radius: 4px;
                }}
            </style>
            
            <div class="code-panels">
                <div class="code-panel">
                    <div class="panel-header">Java Source</div>
                    <div class="code-content" id="{viz_id}_java">
                        {self._format_code_with_lines(java_code)}
                    </div>
                </div>
                
                <div class="code-panel">
                    <div class="panel-header">{target_type.upper()} Target</div>
                    <div class="code-content" id="{viz_id}_target">
                        {self._format_code_with_lines(target_code)}
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
                            const relatedMappings = mappings.filter(m => m.source_line === lineNum);
                            
                            if (relatedMappings.length > 0) {{
                                // Highlight this line
                                line.classList.add('highlighted');
                                
                                // Highlight target lines
                                relatedMappings.forEach(mapping => {{
                                    const targetLine = targetPanel.querySelector(`[data-line="${{mapping.target_line}}"]`);
                                    if (targetLine) {{
                                        targetLine.classList.add('highlighted');
                                        targetLine.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                                    }}
                                }});
                                
                                // Show mapping info
                                infoPanel.innerHTML = relatedMappings.map(m => `
                                    <div>
                                        <strong>${{m.type}}</strong>: ${{m.description}}<br>
                                        Java line ${{m.source_line}} → {target_type} line ${{m.target_line}}
                                    </div>
                                `).join('<hr>');
                            }}
                        }});
                        
                        line.addEventListener('mouseout', () => {{
                            // Remove all highlights
                            document.querySelectorAll('.line').forEach(l => 
                                l.classList.remove('highlighted'));
                            infoPanel.innerHTML = '';
                        }});
                    }});
                    
                    // Add event listeners for target lines
                    targetPanel.querySelectorAll('.line').forEach(line => {{
                        line.addEventListener('mouseover', () => {{
                            const lineNum = parseInt(line.getAttribute('data-line'));
                            const relatedMappings = mappings.filter(m => m.target_line === lineNum);
                            
                            if (relatedMappings.length > 0) {{
                                // Highlight this line
                                line.classList.add('highlighted');
                                
                                // Highlight Java lines
                                relatedMappings.forEach(mapping => {{
                                    const javaLine = javaPanel.querySelector(`[data-line="${{mapping.source_line}}"]`);
                                    if (javaLine) {{
                                        javaLine.classList.add('highlighted');
                                        javaLine.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                                    }}
                                }});
                                
                                // Show mapping info
                                infoPanel.innerHTML = relatedMappings.map(m => `
                                    <div>
                                        <strong>${{m.type}}</strong>: ${{m.description}}<br>
                                        {target_type} line ${{m.target_line}} → Java line ${{m.source_line}}
                                    </div>
                                `).join('<hr>');
                            }}
                        }});
                        
                        line.addEventListener('mouseout', () => {{
                            // Remove all highlights
                            document.querySelectorAll('.line').forEach(l => 
                                l.classList.remove('highlighted'));
                            infoPanel.innerHTML = '';
                        }});
                    }});
                }})();
            </script>
        </div>
        """
        
        return html
    
    def _format_code_with_lines(self, code: str) -> str:
        """Format code with line numbers"""
        lines = code.split('\n')
        return '\n'.join(
            f'<div class="line" data-line="{i+1}">'
            f'<span class="line-number">{i+1}</span>'
            f'<span class="line-content">{line}</span>'
            f'</div>'
            for i, line in enumerate(lines)
        )
    
    def _display_mapping_stats(self, mappings: List[Dict]) -> None:
        """Display statistics about the mappings"""
        if not mappings:
            st.warning("No mappings found between Java and target code.")
            return
        
        # Count mappings by type
        mapping_counts = {}
        for mapping in mappings:
            mapping_type = mapping['type']
            mapping_counts[mapping_type] = mapping_counts.get(mapping_type, 0) + 1
        
        # Display statistics
        st.subheader("Mapping Statistics")
        
        # Create columns for stats
        cols = st.columns(len(mapping_counts))
        for i, (type_name, count) in enumerate(mapping_counts.items()):
            with cols[i]:
                st.metric(
                    label=type_name.replace('_', ' ').title(),
                    value=count
                )
        
        # Display total mappings
        st.metric("Total Mappings", len(mappings))

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