import streamlit as st
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

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
    """Handles visualization using React components"""
    
    def __init__(self):
        self.sample_data = {
            'java': [],
            'ptx': [],
            'opencl': [],
            'mappings': []
        }
    
    def display_code_transition(self, java_blocks: List[CodeBlock],
                              target_blocks: List[CodeBlock],
                              mappings: List[CodeMapping]) -> None:
        """Display code transition visualization using Streamlit components"""
        import streamlit as st
        
        # Create two columns for side-by-side view
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Java Source Code")
            # Display Java blocks with mapping indicators
            for i, block in enumerate(java_blocks):
                # Find mappings for this block
                block_mappings = [m for m in mappings if m.source_id == f"java-{i}"]
                
                # Create a header with mapping info
                header = f"Block {i} (Lines {block.line_start}-{block.line_end})"
                if block_mappings:
                    target_info = [f"→ {m.target_id.split('-')[0].upper()} Block {m.target_id.split('-')[1]} ({m.confidence:.2f})" 
                                 for m in block_mappings]
                    header += f" | Maps to: {', '.join(target_info)}"
                
                with st.expander(header, expanded=True):
                    st.code(block.code, language="java")
        
        with col2:
            # Create tabs for PTX and OpenCL
            ptx_blocks = [b for b in target_blocks if b.language.lower() == 'ptx']
            opencl_blocks = [b for b in target_blocks if b.language.lower() == 'opencl']
            
            if ptx_blocks or opencl_blocks:
                tabs = []
                if ptx_blocks:
                    tabs.append("PTX")
                if opencl_blocks:
                    tabs.append("OpenCL")
                
                active_tab = st.radio("Target Language", tabs)
                
                if active_tab == "PTX":
                    st.markdown("### PTX Target Code")
                    blocks_to_show = ptx_blocks
                else:
                    st.markdown("### OpenCL Target Code")
                    blocks_to_show = opencl_blocks
                
                # Display target blocks with mapping indicators
                for i, block in enumerate(blocks_to_show):
                    # Find mappings for this block
                    block_mappings = [m for m in mappings if m.target_id == f"{block.language.lower()}-{i}"]
                    
                    # Create a header with mapping info
                    header = f"Block {i} (Lines {block.line_start}-{block.line_end})"
                    if block_mappings:
                        source_info = [f"← Java Block {m.source_id.split('-')[1]} ({m.confidence:.2f})" 
                                     for m in block_mappings]
                        header += f" | Maps from: {', '.join(source_info)}"
                    
                    with st.expander(header, expanded=True):
                        st.code(block.code, language=block.language.lower())
        
        # Display mapping details
        st.markdown("### 🔍 Detailed Mappings")
        for mapping in mappings:
            source_block = next((b for i, b in enumerate(java_blocks) 
                               if f"java-{i}" == mapping.source_id), None)
            target_block = next((b for i, b in enumerate(target_blocks) 
                               if f"{b.language.lower()}-{i}" == mapping.target_id), None)
            
            if source_block and target_block:
                with st.expander(
                    f"Java Block {mapping.source_id.split('-')[1]} → "
                    f"{target_block.language.upper()} Block {mapping.target_id.split('-')[1]} "
                    f"(Confidence: {mapping.confidence:.2f})"
                ):
                    for line_map in mapping.line_mapping:
                        st.markdown(f"""
                        - **Type**: {line_map.type}
                        - **Description**: {line_map.description}
                        - **Java Lines**: {', '.join(map(str, line_map.source_lines))}
                        - **{target_block.language.upper()} Lines**: {', '.join(map(str, line_map.target_lines))}
                        ---
                        """)
        
        # Add legend
        st.markdown("""
        ### 📖 Legend
        - **→** indicates Java to target code mapping
        - **←** indicates target to Java code mapping
        - **Confidence** shows how strong the mapping is (0-1)
        """)
    
    def display_performance_analysis(self, blocks: List[CodeBlock]) -> None:
        """Display performance analysis using React"""
        # Group blocks by language
        blocks_by_language = {}
        for block in blocks:
            if block.language not in blocks_by_language:
                blocks_by_language[block.language] = []
            blocks_by_language[block.language].append(block)
        
        # Convert blocks to React format
        performance_data = {}
        for language, blocks in blocks_by_language.items():
            performance_data[language] = [
                {
                    'id': f"{language}-{i}",
                    'code': block.code,
                    'lineStart': block.line_start,
                    'lineEnd': block.line_end,
                    'metrics': block.metrics
                }
                for i, block in enumerate(blocks)
            ]
        
        # Create a container for the React component
        st.markdown("""
        <div id="performance-root"></div>
        <script>
            // Load React and dependencies
            const script = document.createElement('script');
            script.src = 'https://unpkg.com/react@17/umd/react.production.min.js';
            document.head.appendChild(script);
            
            const scriptDOM = document.createElement('script');
            scriptDOM.src = 'https://unpkg.com/react-dom@17/umd/react-dom.production.min.js';
            document.head.appendChild(scriptDOM);
            
            // Load our React component
            const componentScript = document.createElement('script');
            componentScript.src = 'https://your-cdn.com/performance-analyzer.js';
            document.head.appendChild(componentScript);
            
            // Initialize the component with data
            window.addEventListener('load', () => {
                const data = %s;
                const root = ReactDOM.createRoot(document.getElementById('performance-root'));
                root.render(React.createElement(PerformanceAnalyzer, { data }));
            });
        </script>
        """ % json.dumps(performance_data), unsafe_allow_html=True)

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