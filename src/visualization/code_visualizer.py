import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from pygments import highlight
from pygments.lexers import JavaLexer, CLexer
from pygments.formatters import HtmlFormatter
from pygments.styles import get_style_by_name

@dataclass
class CodeBlock:
    """Represents a block of code with metadata"""
    code: str
    language: str
    line_start: int
    line_end: int
    block_type: str
    metrics: Optional[Dict] = None

class CodeVisualizer:
    """Handles visualization of code blocks and their relationships"""
    
    def __init__(self):
        self.style = get_style_by_name('monokai')
        self.formatter = HtmlFormatter(style=self.style)
        self.lexers = {
            'java': JavaLexer(),
            'opencl': CLexer(),  # Using CLexer for OpenCL code
            'ptx': CLexer()  # Using CLexer for PTX as approximation
        }
    
    def display_code_blocks(self, blocks: List[CodeBlock], 
                          show_metrics: bool = True) -> None:
        """Display code blocks with syntax highlighting and metrics"""
        for block in blocks:
            with st.expander(f"{block.language.upper()} Block - Lines {block.line_start}-{block.line_end}"):
                # Syntax highlighted code
                highlighted_code = highlight(
                    block.code,
                    self.lexers.get(block.language.lower(), CLexer()),
                    self.formatter
                )
                st.markdown(
                    f'<style>{self.formatter.get_style_defs()}</style>{highlighted_code}',
                    unsafe_allow_html=True
                )
                
                # Display metrics if available
                if show_metrics and block.metrics:
                    st.markdown("### Performance Metrics")
                    self._display_block_metrics(block.metrics)
    
    def display_code_mapping(self, source_blocks: List[CodeBlock],
                           target_blocks: List[CodeBlock],
                           mappings: List[Tuple[int, int, float]]) -> None:
        """Display interactive code mapping visualization"""
        # Create columns for source and target code
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Source Code")
            source_expanders = {}
            for i, block in enumerate(source_blocks):
                with st.expander(f"Block {i+1} - Lines {block.line_start}-{block.line_end}"):
                    highlighted_code = highlight(
                        block.code,
                        self.lexers.get(block.language.lower(), CLexer()),
                        self.formatter
                    )
                    source_expanders[i] = st.markdown(
                        f'<style>{self.formatter.get_style_defs()}</style>{highlighted_code}',
                        unsafe_allow_html=True
                    )
        
        with col2:
            st.markdown("### Target Code")
            target_expanders = {}
            for i, block in enumerate(target_blocks):
                with st.expander(f"Block {i+1} - Lines {block.line_start}-{block.line_end}"):
                    highlighted_code = highlight(
                        block.code,
                        self.lexers.get(block.language.lower(), CLexer()),
                        self.formatter
                    )
                    target_expanders[i] = st.markdown(
                        f'<style>{self.formatter.get_style_defs()}</style>{highlighted_code}',
                        unsafe_allow_html=True
                    )
        
        # Display mapping visualization
        st.markdown("### Code Mapping Visualization")
        self._create_mapping_visualization(source_blocks, target_blocks, mappings)
    
    def display_performance_analysis(self, blocks: List[CodeBlock]) -> None:
        """Display performance analysis visualizations"""
        # Group blocks by language
        blocks_by_language = {}
        for block in blocks:
            if block.language not in blocks_by_language:
                blocks_by_language[block.language] = []
            blocks_by_language[block.language].append(block)
        
        # Display metrics for each language
        for language, blocks in blocks_by_language.items():
            st.markdown(f"### {language.upper()} Performance Analysis")
            
            # Create tabs for different metrics
            metric_tabs = st.tabs([
                "Instruction Mix",
                "Memory Access",
                "Thread Divergence"
            ])
            
            with metric_tabs[0]:
                self._display_instruction_mix(blocks)
            
            with metric_tabs[1]:
                self._display_memory_access(blocks)
            
            with metric_tabs[2]:
                self._display_thread_divergence(blocks)
    
    def display_control_flow(self, blocks: List[CodeBlock], 
                           edges: List[Tuple[int, int]]) -> None:
        """Display interactive control flow visualization"""
        st.markdown("## Control Flow Analysis")
        
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes with metadata
        for i, block in enumerate(blocks):
            G.add_node(i, 
                      code=block.code[:50] + "..." if len(block.code) > 50 else block.code,
                      language=block.language,
                      lines=f"{block.line_start}-{block.line_end}")
        
        # Add edges
        G.add_edges_from(edges)
        
        # Create positions for nodes
        pos = nx.spring_layout(G)
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(
                f"Block {node}<br>"
                f"Language: {G.nodes[node]['language']}<br>"
                f"Lines: {G.nodes[node]['lines']}<br>"
                f"Code: {G.nodes[node]['code']}"
            )
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[f"Block {i}" for i in range(len(node_x))],
            hovertext=node_text,
            marker=dict(
                size=30,
                color=['#1f77b4' if G.nodes[node]['language'] == 'java' else
                       '#2ca02c' if G.nodes[node]['language'] == 'opencl' else
                       '#ff7f0e' for node in G.nodes()],
                line=dict(width=2)
            )
        )
        
        # Create edge trace
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_block_metrics(self, metrics: Dict) -> None:
        """Display metrics for a code block"""
        # Convert metrics to DataFrame for better display
        metrics_data = []
        for category, values in metrics.items():
            if isinstance(values, dict):
                for metric, value in values.items():
                    metrics_data.append({
                        'Category': category,
                        'Metric': metric,
                        'Value': value if isinstance(value, (int, float)) else len(value)
                    })
            else:
                metrics_data.append({
                    'Category': 'General',
                    'Metric': category,
                    'Value': values if isinstance(values, (int, float)) else len(values)
                })
        
        df = pd.DataFrame(metrics_data)
        if not df.empty:
            st.dataframe(df)
    
    def _create_mapping_visualization(self, source_blocks: List[CodeBlock],
                                    target_blocks: List[CodeBlock],
                                    mappings: List[Tuple[int, int, float]]) -> None:
        """Create interactive visualization of code mappings"""
        # Create Sankey diagram
        source_indices = []
        target_indices = []
        values = []
        labels = []
        
        # Add source nodes
        for i, block in enumerate(source_blocks):
            labels.append(f"Source {i+1}\n{block.language}")
        
        # Add target nodes
        target_offset = len(source_blocks)
        for i, block in enumerate(target_blocks):
            labels.append(f"Target {i+1}\n{block.language}")
        
        # Add links
        for source, target, weight in mappings:
            source_indices.append(source)
            target_indices.append(target + target_offset)
            values.append(weight * 100)  # Scale weight to percentage
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=["#1f77b4"] * len(source_blocks) + 
                      ["#2ca02c"] * len(target_blocks)
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values
            )
        )])
        
        fig.update_layout(title_text="Code Block Mappings", font_size=10)
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_instruction_mix(self, blocks: List[CodeBlock]) -> None:
        """Display instruction mix analysis"""
        # Collect instruction mix data
        instruction_data = []
        for block in blocks:
            if block.metrics and isinstance(block.metrics, dict) and 'instruction_mix' in block.metrics:
                mix = block.metrics['instruction_mix']
                if isinstance(mix, dict):
                    instruction_data.append({
                        'Block': f"Block {block.line_start}-{block.line_end}",
                        'Arithmetic': mix.get('arithmetic', 0),
                        'Memory': mix.get('memory', 0),
                        'Control': mix.get('control', 0),
                        'Conversion': mix.get('conversion', 0),
                        'Special': mix.get('special', 0)
                    })
        
        if instruction_data:
            df = pd.DataFrame(instruction_data)
            fig = px.bar(df, x='Block', y=['Arithmetic', 'Memory', 'Control', 'Conversion', 'Special'],
                        title='Instruction Mix by Block',
                        barmode='stack')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No instruction mix data available")
    
    def _display_memory_access(self, blocks: List[CodeBlock]) -> None:
        """Display memory access patterns"""
        # Collect memory access data
        memory_data = []
        for block in blocks:
            if block.metrics and isinstance(block.metrics, dict):
                if 'memory_operations' in block.metrics:
                    ops = block.metrics['memory_operations']
                    if isinstance(ops, dict):
                        memory_data.append({
                            'Block': f"Block {block.line_start}-{block.line_end}",
                            'Global Loads': len(ops.get('global_loads', [])),
                            'Global Stores': len(ops.get('global_stores', [])),
                            'Shared Memory': len(ops.get('shared_memory', [])),
                            'Atomic': len(ops.get('atomic', []))
                        })
                elif 'memory_patterns' in block.metrics:
                    patterns = block.metrics['memory_patterns']
                    if isinstance(patterns, dict):
                        memory_data.append({
                            'Block': f"Block {block.line_start}-{block.line_end}",
                            'Global Reads': len(patterns.get('global_reads', [])),
                            'Global Writes': len(patterns.get('global_writes', [])),
                            'Local Memory': len(patterns.get('local_memory', [])),
                            'Private Memory': len(patterns.get('private_memory', []))
                        })
        
        if memory_data:
            df = pd.DataFrame(memory_data)
            fig = px.bar(df, x='Block', y=[col for col in df.columns if col != 'Block'],
                        title='Memory Access Patterns by Block',
                        barmode='stack')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No memory access data available")
    
    def _display_thread_divergence(self, blocks: List[CodeBlock]) -> None:
        """Display thread divergence analysis"""
        # Collect thread divergence data
        divergence_data = []
        for block in blocks:
            if block.metrics and isinstance(block.metrics, dict) and 'thread_divergence' in block.metrics:
                divergence = block.metrics['thread_divergence']
                if isinstance(divergence, dict):
                    divergence_data.append({
                        'Block': f"Block {block.line_start}-{block.line_end}",
                        'Conditional Branches': len(divergence.get('conditional_branches', [])),
                        'Loop Divergence': len(divergence.get('loop_divergence', [])),
                        'Atomic Operations': len(divergence.get('atomic_operations', []))
                    })
        
        if divergence_data:
            df = pd.DataFrame(divergence_data)
            fig = px.bar(df, x='Block', y=['Conditional Branches', 'Loop Divergence', 'Atomic Operations'],
                        title='Thread Divergence Analysis by Block',
                        barmode='stack')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No thread divergence data available") 