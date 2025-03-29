import streamlit as st
import sys
import os
from src.analysis.semantic_analysis import SemanticAnalyzer
from src.analysis.performance_analysis import PerformanceAnalyzer
from src.visualization.react_visualizer import ReactVisualizer
from src.visualization.code_visualizer import CodeBlock

class TornadoCodeTransitionVisualizer:
    """Main class for visualizing TornadoVM code transitions"""
    
    def __init__(self):
        self.semantic_analyzer = SemanticAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()
        self.visualizer = ReactVisualizer()
    
    def parse_java_code(self, code: str) -> list[CodeBlock]:
        """Parse Java code into blocks"""
        blocks = []
        current_block = []
        current_line = 0
        
        for line in code.split('\n'):
            current_line += 1
            current_block.append(line)
            
            # End block on method end or class end
            if line.strip() == '}' and len(current_block) > 1:
                block_code = '\n'.join(current_block)
                blocks.append(CodeBlock(
                    code=block_code,
                    language='java',
                    line_start=current_line - len(current_block) + 1,
                    line_end=current_line,
                    block_type='method'
                ))
                current_block = []
        
        return blocks
    
    def parse_opencl_code(self, code: str) -> list[CodeBlock]:
        """Parse OpenCL code into blocks"""
        blocks = []
        current_block = []
        current_line = 0
        
        for line in code.split('\n'):
            current_line += 1
            current_block.append(line)
            
            # End block on kernel end
            if line.strip() == '}' and len(current_block) > 1:
                block_code = '\n'.join(current_block)
                blocks.append(CodeBlock(
                    code=block_code,
                    language='opencl',
                    line_start=current_line - len(current_block) + 1,
                    line_end=current_line,
                    block_type='kernel'
                ))
                current_block = []
        
        return blocks
    
    def parse_ptx_code(self, code: str) -> list[CodeBlock]:
        """Parse PTX code into blocks"""
        blocks = []
        current_block = []
        current_line = 0
        
        for line in code.split('\n'):
            current_line += 1
            current_block.append(line)
            
            # End block on label or ret
            if line.strip().startswith('BLOCK_') or line.strip() == 'ret;':
                block_code = '\n'.join(current_block)
                blocks.append(CodeBlock(
                    code=block_code,
                    language='ptx',
                    line_start=current_line - len(current_block) + 1,
                    line_end=current_line,
                    block_type='kernel'
                ))
                current_block = []
        
        return blocks
    
    def analyze_code_mappings(self, java_blocks: list[CodeBlock],
                            target_blocks: list[CodeBlock]) -> list:
        """Analyze mappings between Java and target code blocks"""
        mappings = []
        
        # Determine target type from first block
        if not target_blocks:
            return mappings
            
        target_type = target_blocks[0].language.lower()
        
        for java_block in java_blocks:
            for target_block in target_blocks:
                # Analyze semantic mappings
                semantic_mappings = self.semantic_analyzer.analyze_code_mapping(
                    java_block.code,
                    target_block.code,
                    target_type
                )
                
                if semantic_mappings:
                    # Calculate similarity score based on number of matching patterns
                    similarity_score = len(semantic_mappings) / max(
                        len(java_block.code.split('\n')),
                        len(target_block.code.split('\n'))
                    )
                    
                    if similarity_score >= 0.3:  # Threshold for meaningful mappings
                        mappings.append({
                            'source_id': f"java-{java_blocks.index(java_block)}",
                            'target_id': f"{target_type}-{target_blocks.index(target_block)}",
                            'type': 'method-to-kernel',
                            'confidence': similarity_score,
                            'line_mapping': semantic_mappings
                        })
        
        return mappings
    
    def create_control_flow_edges(self, blocks: list[CodeBlock]) -> list:
        """Create control flow edges between blocks"""
        edges = []
        
        for i, block in enumerate(blocks):
            # Find patterns in the block
            patterns = self.semantic_analyzer.find_patterns(
                block.code,
                block.language.lower()
            )
            
            # Create edges based on patterns
            for pattern in patterns:
                if pattern['pattern_type'] in ['parallel_loop', 'sequential_loop']:
                    # Connect to next block
                    if i < len(blocks) - 1:
                        edges.append((i, i + 1))
                elif pattern['pattern_type'] == 'conditional_branch':
                    # Connect to next block and branch target
                    if i < len(blocks) - 1:
                        edges.append((i, i + 1))
                        # Find branch target block
                        for j, target_block in enumerate(blocks[i+1:], i+1):
                            if self.semantic_analyzer.find_patterns(
                                target_block.code,
                                target_block.language.lower()
                            ):
                                edges.append((i, j))
                                break
        
        return edges

def main():
    st.set_page_config(
        page_title="TornadoVM Code Transition Visualizer",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("TornadoVM Code Transition Visualizer")
    st.markdown("""
    This tool helps visualize and analyze how Java code is transformed into PTX or OpenCL code
    by TornadoVM. Upload your source and target code files to get started.
    """)
    
    # File upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Java Source Code")
        java_file = st.file_uploader("Upload Java file", type=['java'])
        java_code = java_file.read().decode() if java_file else None
    
    with col2:
        st.subheader("Target Code")
        target_type = st.radio("Select target type", ["PTX", "OpenCL"])
        target_file = st.file_uploader(
            f"Upload {target_type} file",
            type=['ptx', 'cl'] if target_type == "PTX" else ['cl']
        )
        target_code = target_file.read().decode() if target_file else None
    
    if java_code and target_code:
        visualizer = TornadoCodeTransitionVisualizer()
        
        # Parse code into blocks
        java_blocks = visualizer.parse_java_code(java_code)
        target_blocks = (
            visualizer.parse_ptx_code(target_code)
            if target_type == "PTX"
            else visualizer.parse_opencl_code(target_code)
        )
        
        # Analyze code mappings
        mappings = visualizer.analyze_code_mappings(java_blocks, target_blocks)
        
        # Create control flow edges
        edges = visualizer.create_control_flow_edges(java_blocks + target_blocks)
        
        # Create tabs for different views
        tabs = st.tabs([
            "Code Mapping",
            "Control Flow",
            "Performance Analysis"
        ])
        
        with tabs[0]:
            # Display code mapping visualization
            visualizer.visualizer.display_code_transition(
                java_blocks,
                target_blocks,
                mappings
            )
        
        with tabs[1]:
            # Display control flow visualization
            visualizer.visualizer.display_control_flow(
                java_blocks + target_blocks,
                edges
            )
        
        with tabs[2]:
            # Display performance analysis
            visualizer.visualizer.display_performance_analysis(
                target_blocks
            )
    else:
        st.info("Please upload both Java and target code files to begin visualization.")

if __name__ == "__main__":
    main() 