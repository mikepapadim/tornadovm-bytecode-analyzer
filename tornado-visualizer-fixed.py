import re
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
import pandas as pd
from collections import defaultdict
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="TornadoVM Bytecode Visualizer",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Data Classes for representing the TornadoVM bytecode structure
@dataclass
class BytecodeOperation:
    """Represents a single bytecode operation"""
    operation: str  # e.g., ALLOC, TRANSFER_HOST_TO_DEVICE, etc.
    objects: List[str] = field(default_factory=list)  # Object references
    size: int = 0
    batch_size: int = 0
    task_name: str = ""
    event_list: int = -1
    offset: int = 0
    status: str = ""  # For DEALLOC status (Persisted/Freed)

@dataclass
class MemoryObject:
    """Tracks a memory object through its lifecycle"""
    object_id: str  # Hash ID
    object_type: str
    size: int = 0
    allocated_in_graph: str = ""
    current_status: str = "Unknown"  # Allocated, Transferred, Persisted, Freed
    allocation_op_index: int = -1
    deallocation_op_index: int = -1
    used_in_graphs: Set[str] = field(default_factory=set)
    transfer_history: List[Tuple[str, str, int]] = field(default_factory=list)  # (type, graph_id, op_index)

@dataclass
class TaskGraph:
    """Represents a TornadoVM task graph"""
    graph_id: str  # Extracted from the log
    device: str
    thread: str
    operations: List[BytecodeOperation] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))  # Graph -> [objects]
    objects_produced: Set[str] = field(default_factory=set)  # Objects created or modified
    objects_consumed: Set[str] = field(default_factory=set)  # Objects used but not created
    tasks: List[str] = field(default_factory=list)  # Named tasks in this graph

class TornadoVisualizer:
    """Main class for parsing and visualizing TornadoVM bytecode logs"""
    
    def __init__(self):
        self.task_graphs = []
        self.memory_objects = {}
        self.dependency_graph = nx.DiGraph()
        self.bytecode_details = []  # For detailed bytecode visualization
        
    def parse_log(self, log_content: str) -> None:
        """Parse the TornadoVM bytecode log and extract task graphs"""
        # Split the log into sections for each task graph
        pattern = r"Interpreter instance running bytecodes for:(.*?)bc:\s+END"
        graph_sections = re.findall(pattern, log_content, re.DOTALL)
        
        for i, section in enumerate(graph_sections):
            graph_name = f"TaskGraph_{i}"
            # Try to extract graph name from task names if possible
            task_match = re.search(r"task ([\w\.]+)\.", section)
            if task_match:
                graph_name = task_match.group(1)
                
            self._parse_task_graph(section, graph_name)
            
        # Build dependencies after all graphs are parsed
        self._build_dependencies()
        
    def _parse_task_graph(self, section: str, graph_id: str) -> None:
        """Parse a single task graph section"""
        # Extract device and thread info
        device_match = re.search(r"PTX -- (.*?) Running in thread:\s+(.*?)$", section, re.MULTILINE)
        if not device_match:
            return
            
        device = device_match.group(1)
        thread = device_match.group(2).strip()
        
        task_graph = TaskGraph(graph_id=graph_id, device=device, thread=thread)
        
        # Parse bytecode operations
        bc_pattern = r"bc:\s+(\w+)\s+(.*?)$"
        global_op_index = sum(len(g.operations) for g in self.task_graphs)
        
        # Track tasks in this graph
        tasks = set()
        
        for op_match in re.finditer(bc_pattern, section, re.MULTILINE):
            op_type = op_match.group(1)
            op_details = op_match.group(2)
            
            # Create and add the operation
            operation = self._parse_operation(op_type, op_details)
            task_graph.operations.append(operation)
            
            # Track task names from LAUNCH operations
            if op_type == "LAUNCH" and operation.task_name:
                tasks.add(operation.task_name)
            
            # Store bytecode details for visualization
            self.bytecode_details.append({
                "TaskGraph": graph_id,
                "Operation": op_type,
                "Details": op_details,
                "GlobalIndex": global_op_index,
                "Objects": ", ".join(operation.objects),
                "TaskName": operation.task_name
            })
            global_op_index += 1
            
            # Track objects and tasks
            self._process_operation(operation, task_graph)
        
        # Add discovered tasks to the graph
        task_graph.tasks = list(tasks) if tasks else [f"{graph_id}_main"]
            
        self.task_graphs.append(task_graph)
    
    def _parse_operation(self, op_type: str, op_details: str) -> BytecodeOperation:
        """Parse a single bytecode operation"""
        operation = BytecodeOperation(operation=op_type)
        
        if op_type == "ALLOC":
            # Extract object reference and size
            obj_match = re.search(r"([\w\.]+@[0-9a-f]+) on\s+.*?, size=(\d+), batchSize=(\d+)", op_details)
            if obj_match:
                operation.objects.append(obj_match.group(1))
                operation.size = int(obj_match.group(2))
                operation.batch_size = int(obj_match.group(3))
                
        elif op_type.startswith("TRANSFER"):
            # Extract object reference and size
            obj_match = re.search(r"\[(0x[0-9a-f]+|Object Hash Code=0x[0-9a-f]+)\] ([\w\.]+@[0-9a-f]+) on\s+.*?, size=(\d+), batchSize=(\d+)", op_details)
            if obj_match:
                operation.objects.append(obj_match.group(2))
                operation.size = int(obj_match.group(3))
                operation.batch_size = int(obj_match.group(4))
                
                # Extract event list if present
                event_match = re.search(r"\[event list=(-?\d+)\]", op_details)
                if event_match:
                    operation.event_list = int(event_match.group(1))
                    
        elif op_type == "LAUNCH":
            # Extract task name - modified to capture the full task name
            task_match = re.search(r"task ([\w\.]+) - ([\w\.]+) on", op_details)
            if task_match:
                operation.task_name = task_match.group(1)  # Just use the main task name
                
                # Extract event list if present
                event_match = re.search(r"\[event list=(\d+)\]", op_details)
                if event_match:
                    operation.event_list = int(event_match.group(1))
                    
        elif op_type == "DEALLOC":
            # Extract object reference and status
            obj_match = re.search(r"\[(0x[0-9a-f]+)\] ([\w\.]+@[0-9a-f]+) \[Status:\s+([\w\s]+)\]", op_details)
            if obj_match:
                operation.objects.append(obj_match.group(2))
                operation.status = obj_match.group(3).strip()
                
        elif op_type == "ON_DEVICE_BUFFER" or op_type == "ON_DEVICE":
            # Extract object reference
            obj_match = re.search(r"\[(0x[0-9a-f]+)\] ([\w\.]+@[0-9a-f]+)", op_details)
            if obj_match:
                operation.objects.append(obj_match.group(2))
                
        elif op_type == "BARRIER":
            # Extract event list
            event_match = re.search(r"event-list (\d+)", op_details)
            if event_match:
                operation.event_list = int(event_match.group(1))
                
        return operation
    
    def _process_operation(self, operation: BytecodeOperation, task_graph: TaskGraph) -> None:
        """Process an operation to track objects and tasks"""
        op_type = operation.operation
        
        # Track objects
        for obj_ref in operation.objects:
            obj_hash = self._extract_hash(obj_ref)
            obj_type = self._extract_type(obj_ref)
            
            if op_type == "ALLOC":
                # Create or update memory object
                if obj_hash not in self.memory_objects:
                    self.memory_objects[obj_hash] = MemoryObject(
                        object_id=obj_hash,
                        object_type=obj_type,
                        size=operation.size,
                        allocated_in_graph=task_graph.graph_id,
                        current_status="Allocated",
                        allocation_op_index=len(task_graph.operations) - 1
                    )
                    task_graph.objects_produced.add(obj_hash)
                
            elif op_type.startswith("TRANSFER"):
                # Track transfer
                if obj_hash in self.memory_objects:
                    self.memory_objects[obj_hash].transfer_history.append(
                        (op_type, task_graph.graph_id, len(task_graph.operations) - 1)
                    )
                    self.memory_objects[obj_hash].current_status = "Transferred"
                    self.memory_objects[obj_hash].used_in_graphs.add(task_graph.graph_id)
                    
                    if op_type.startswith("TRANSFER_HOST_TO_DEVICE"):
                        task_graph.objects_consumed.add(obj_hash)
                    
            elif op_type == "DEALLOC":
                # Update deallocation info
                if obj_hash in self.memory_objects:
                    self.memory_objects[obj_hash].current_status = f"Deallocated ({operation.status})"
                    self.memory_objects[obj_hash].deallocation_op_index = len(task_graph.operations) - 1
                    
                    # If persisted, this object can be used by future graphs
                    if "Persisted" in operation.status:
                        task_graph.objects_produced.add(obj_hash)
                    
            elif op_type == "ON_DEVICE_BUFFER" or op_type == "ON_DEVICE":
                # Object is being reused from a previous graph
                if obj_hash in self.memory_objects:
                    self.memory_objects[obj_hash].used_in_graphs.add(task_graph.graph_id)
                    task_graph.objects_consumed.add(obj_hash)
                    
                    # Find where the object was produced
                    for prev_graph in self.task_graphs:
                        if obj_hash in prev_graph.objects_produced:
                            task_graph.dependencies[prev_graph.graph_id].append(obj_hash)
                    
            elif op_type == "LAUNCH" and operation.task_name:
                # Track task
                task_graph.tasks.append(operation.task_name)
    
    def _extract_hash(self, obj_ref: str) -> str:
        """Extract hash from object reference"""
        match = re.search(r"@([0-9a-f]+)", obj_ref)
        return match.group(1) if match else obj_ref
    
    def _extract_type(self, obj_ref: str) -> str:
        """Extract type from object reference"""
        match = re.search(r"([\w\.]+)@", obj_ref)
        return match.group(1) if match else "Unknown"
    
    def _build_dependencies(self) -> None:
        """Build dependencies between task graphs based on object usage"""
        # Create nodes for each task graph
        for graph in self.task_graphs:
            label = f"{graph.graph_id}\n({len(graph.tasks)} tasks)"
            self.dependency_graph.add_node(graph.graph_id, label=label, 
                                          tasks=', '.join(graph.tasks),
                                          num_operations=len(graph.operations))
        
        # Create edges for dependencies
        for graph in self.task_graphs:
            for producer_graph, objects in graph.dependencies.items():
                # Create a unique edge with objects as labels
                self.dependency_graph.add_edge(
                    producer_graph, 
                    graph.graph_id, 
                    objects=objects,
                    label='\n'.join([obj[:8] for obj in objects[:3]]) +  
                          (f"\n+{len(objects)-3} more" if len(objects) > 3 else "")
                )
    
    def visualize_dependency_graph_detailed(self) -> Optional[plt.Figure]:
        """Visualize the task graph dependencies with detailed bytecode information"""
        if not self.task_graphs:
            return None
            
        fig, ax = plt.subplots(figsize=(14, 10))
        
        try:
            # Use a hierarchical layout for better flow visualization
            pos = nx.spring_layout(self.dependency_graph)  # Fallback to spring layout
            
            # Draw nodes with custom appearance
            node_sizes = [3000 + (self.dependency_graph.nodes[n].get('num_operations', 0) * 100) 
                         for n in self.dependency_graph.nodes()]
            
            # Create a colormap for nodes
            node_colors = []
            for i, n in enumerate(self.dependency_graph.nodes()):
                # Use different colors for different task graphs
                color = plt.cm.tab10(i % 10)
                # Make it semi-transparent
                node_colors.append(color)
            
            # Draw nodes
            nx.draw_networkx_nodes(self.dependency_graph, pos, node_size=node_sizes, 
                                  node_color=node_colors, alpha=0.8, ax=ax)
            
            # Draw edges with custom appearance
            edge_colors = []
            edge_widths = []
            edge_styles = []
            
            for u, v, data in self.dependency_graph.edges(data=True):
                # Edge color based on object type
                num_objs = len(data.get('objects', []))
                edge_colors.append('darkorange' if num_objs > 0 else 'gray')
                edge_widths.append(1 + min(num_objs, 5))  # Thicker for more objects
                edge_styles.append('solid')
            
            nx.draw_networkx_edges(self.dependency_graph, pos, edge_color=edge_colors, 
                                  width=edge_widths, style=edge_styles, 
                                  arrowsize=20, arrowstyle='->', ax=ax)
            
            # Add node labels
            node_labels = {n: self.dependency_graph.nodes[n].get('label', n) for n in self.dependency_graph.nodes()}
            nx.draw_networkx_labels(self.dependency_graph, pos, labels=node_labels, 
                                   font_size=12, font_weight='bold', ax=ax)
            
            # Add edge labels showing which objects are passed between graphs
            edge_labels = {(u, v): data.get('label', '') for u, v, data in self.dependency_graph.edges(data=True)}
            nx.draw_networkx_edge_labels(self.dependency_graph, pos, edge_labels=edge_labels, 
                                        font_size=9, ax=ax)
            
            # Add bytecode details for each task graph as a text box
            for i, graph in enumerate(self.task_graphs):
                if graph.graph_id in pos:
                    x, y = pos[graph.graph_id]
                    text_x = x + 0.2  # Adjust position
                    
                    # Create a text with bytecode operations summary
                    op_counts = defaultdict(int)
                    for op in graph.operations:
                        op_counts[op.operation] += 1
                    
                    bytecode_text = f"Bytecodes:\n"
                    for op, count in op_counts.items():
                        bytecode_text += f"{op}: {count}\n"
                    
                    # Add text box with bytecode summary
                    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
                    ax.text(text_x, y, bytecode_text, fontsize=8,
                            verticalalignment='center', bbox=props)
            
            plt.title("Task Graph Dependencies with Bytecode Information", fontsize=16)
            plt.axis("off")
            return fig
        except Exception as e:
            st.error(f"Error generating dependency graph: {e}")
            return None
    
    def visualize_simple_dependency_graph(self) -> Optional[plt.Figure]:
        """Creates a simpler dependency graph visualization"""
        if not self.task_graphs:
            return None
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        try:
            # Simple layout
            pos = nx.spring_layout(self.dependency_graph, seed=42)
            
            # Draw nodes
            nx.draw_networkx_nodes(self.dependency_graph, pos, 
                                  node_size=2000, 
                                  node_color='skyblue', 
                                  ax=ax)
            
            # Draw edges
            nx.draw_networkx_edges(self.dependency_graph, pos, 
                                  width=2, 
                                  edge_color='gray',
                                  arrowsize=20, 
                                  arrowstyle='->', 
                                  ax=ax)
            
            # Add labels
            nx.draw_networkx_labels(self.dependency_graph, pos, font_size=12, ax=ax)
            
            plt.title("Task Graph Dependencies", fontsize=16)
            plt.axis("off")
            return fig
        except Exception as e:
            st.error(f"Error generating simple dependency graph: {e}")
            return None
    
    def visualize_memory_timeline_interactive(self) -> go.Figure:
        """Create an enhanced interactive timeline of memory operations"""
        # Prepare data
        operations = []
        
        for i, graph in enumerate(self.task_graphs):
            for j, op in enumerate(graph.operations):
                if op.operation in ["ALLOC", "TRANSFER_HOST_TO_DEVICE_ONCE", "TRANSFER_HOST_TO_DEVICE_ALWAYS", 
                                  "TRANSFER_DEVICE_TO_HOST_ALWAYS", "DEALLOC", "ON_DEVICE", "ON_DEVICE_BUFFER"]:
                    for obj_ref in operation.objects:
                        obj_hash = self._extract_hash(obj_ref)
                        obj_type = self._extract_type(obj_ref)
                        operations.append({
                            "TaskGraph": graph.graph_id,
                            "Operation": op.operation,
                            "Object": obj_hash,
                            "ObjectType": obj_type,
                            "Size": op.size,
                            "OperationIndex": j,
                            "GlobalIndex": sum(len(g.operations) for g in self.task_graphs[:i]) + j,
                            "Status": op.status if op.operation == "DEALLOC" else ""
                        })
        
        df = pd.DataFrame(operations)
        if df.empty:
            return go.Figure()
            
        # Create a more sophisticated timeline
        fig = go.Figure()
        
        # Color mapping for operations
        color_map = {
            "ALLOC": "#22c55e",  # Green
            "TRANSFER_HOST_TO_DEVICE_ONCE": "#3b82f6",  # Blue
            "TRANSFER_HOST_TO_DEVICE_ALWAYS": "#1d4ed8",  # Dark blue
            "TRANSFER_DEVICE_TO_HOST_ALWAYS": "#8b5cf6",  # Purple
            "DEALLOC": "#ef4444",  # Red
            "ON_DEVICE": "#f97316",  # Orange
            "ON_DEVICE_BUFFER": "#fb923c"  # Light orange
        }
        
        # Add traces for each operation type
        for op_type, color in color_map.items():
            df_filtered = df[df["Operation"] == op_type]
            if not df_filtered.empty:
                # Size mapping for markers - make it proportional to data size but with min/max constraints
                size_ref = df_filtered["Size"].max() if not df_filtered.empty else 1
                sizes = df_filtered["Size"].apply(lambda x: max(8, min(20, 8 + (x / size_ref) * 12)))
                
                fig.add_trace(go.Scatter(
                    x=df_filtered["GlobalIndex"],
                    y=df_filtered["Object"],
                    mode="markers",
                    marker=dict(
                        color=color, 
                        size=sizes,
                        line=dict(width=1, color='DarkSlateGrey')
                    ),
                    name=op_type,
                    text=df_filtered.apply(
                        lambda row: f"<b>{row['Operation']}</b> in {row['TaskGraph']}<br>"
                                   f"Object: {row['ObjectType']}@{row['Object']}<br>"
                                   f"Size: {row['Size']:,} bytes" + 
                                   (f"<br>Status: {row['Status']}" if row['Status'] else ""),
                        axis=1
                    ),
                    hoverinfo="text"
                ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': "Memory Operations Timeline",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24}
            },
            xaxis_title="Operation Sequence",
            yaxis_title="Memory Objects",
            height=600,
            template="plotly_white",
        )
        
        return fig
    
    def visualize_object_flow(self, selected_object: Optional[str] = None) -> go.Figure:
        """Visualize the flow of a specific object through task graphs"""
        if not selected_object and self.memory_objects:
            # Choose the first object if none specified
            selected_object = next(iter(self.memory_objects.keys()))
            
        if not selected_object or selected_object not in self.memory_objects:
            return go.Figure()
            
        obj = self.memory_objects[selected_object]
        
        # Prepare data
        events = []
        
        # Add allocation event
        alloc_graph = obj.allocated_in_graph
        events.append({
            "TaskGraph": alloc_graph,
            "Event": "Allocated",
            "Size": obj.size,
            "EventIndex": obj.allocation_op_index,
            "GlobalIndex": sum(len(g.operations) for g in self.task_graphs 
                              if g.graph_id < alloc_graph) + obj.allocation_op_index
        })
        
        # Add transfer events
        for transfer_type, graph_id, op_index in obj.transfer_history:
            global_index = sum(len(g.operations) for g in self.task_graphs 
                              if g.graph_id < graph_id) + op_index
            events.append({
                "TaskGraph": graph_id,
                "Event": transfer_type,
                "Size": obj.size,
                "EventIndex": op_index,
                "GlobalIndex": global_index
            })
        
        # Add deallocation event if present
        if obj.deallocation_op_index >= 0:
            dealloc_graph = None
            for graph in self.task_graphs:
                if obj.object_id in [self._extract_hash(obj_ref) for op in graph.operations 
                                    for obj_ref in op.objects if op.operation == "DEALLOC"]:
                    dealloc_graph = graph.graph_id
                    break
                    
            if dealloc_graph:
                global_index = sum(len(g.operations) for g in self.task_graphs 
                                  if g.graph_id < dealloc_graph) + obj.deallocation_op_index
                events.append({
                    "TaskGraph": dealloc_graph,
                    "Event": f"Deallocated ({obj.current_status})",
                    "Size": obj.size,
                    "EventIndex": obj.deallocation_op_index,
                    "GlobalIndex": global_index
                })
        
        df = pd.DataFrame(events)
        if df.empty:
            return go.Figure()
        
        # Create enhanced flow visualization
        fig = go.Figure()
        
        # Color mapping for events
        color_map = {
            "Allocated": "#22c55e",  # Green
            "TRANSFER_HOST_TO_DEVICE_ONCE": "#3b82f6",  # Blue
            "TRANSFER_HOST_TO_DEVICE_ALWAYS": "#1d4ed8",  # Dark blue
            "TRANSFER_DEVICE_TO_HOST_ALWAYS": "#8b5cf6",  # Purple
        }
        
        # Add line connecting events
        fig.add_trace(go.Scatter(
            x=df["GlobalIndex"],
            y=[1] * len(df),
            mode="lines",
            line=dict(color="rgba(0,0,0,0.3)", width=3),
            hoverinfo="none",
            showlegend=False
        ))
        
        # Add markers for each event
        for _, row in df.iterrows():
            event_type = row["Event"]
            if event_type.startswith("Deallocated"):
                color = "#ef4444"  # Red
                symbol = "x"
                size = 15
            else:
                color = color_map.get(event_type, "gray")
                symbol = "circle"
                size = 15
                
            fig.add_trace(go.Scatter(
                x=[row["GlobalIndex"]],
                y=[1],
                mode="markers",
                marker=dict(
                    color=color, 
                    size=size, 
                    symbol=symbol,
                    line=dict(width=1, color='black')
                ),
                name=event_type,
                text=f"<b>{event_type}</b> in {row['TaskGraph']}<br>Size: {row['Size']:,} bytes",
                hoverinfo="text"
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': f"Object Flow: {obj.object_type}@{obj.object_id}",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 18}
            },
            xaxis_title="Operation Sequence",
            yaxis=dict(
                showticklabels=False,
                zeroline=False,
                showgrid=False,
                range=[0.5, 1.8]
            ),
            hovermode="closest",
            height=300,
            width=1000,
            template="plotly_white",
            margin=dict(l=50, r=50, t=80, b=50),
            showlegend=True
        )
        
        return fig
    
    def get_detailed_bytecode_view(self) -> pd.DataFrame:
        """Get a detailed view of all bytecode operations"""
        return pd.DataFrame(self.bytecode_details)
    
    def generate_task_summary(self) -> pd.DataFrame:
        """Generate a summary of tasks and memory operations"""
        task_data = []
        
        for graph in self.task_graphs:
            # Base metrics for the graph
            num_allocs = sum(1 for op in graph.operations if op.operation == "ALLOC")
            num_transfers = sum(1 for op in graph.operations 
                               if op.operation.startswith("TRANSFER"))
            num_deallocs = sum(1 for op in graph.operations if op.operation == "DEALLOC")
            num_persisted = sum(1 for op in graph.operations 
                               if op.operation == "DEALLOC" and "Persisted" in op.status)
            
            # Calculate total memory allocated/transferred
            mem_allocated = sum(op.size for op in graph.operations if op.operation == "ALLOC")
            mem_transferred = sum(op.size for op in graph.operations 
                                if op.operation.startswith("TRANSFER"))
            
            # Get exact object dependencies
            dep_details = []
            for dep_graph, obj_hashes in graph.dependencies.items():
                obj_details = []
                for obj_hash in obj_hashes:
                    if obj_hash in self.memory_objects:
                        obj = self.memory_objects[obj_hash]
                        obj_details.append(f"{obj.object_type}@{obj_hash}")
                if obj_details:
                    dep_details.append(f"{dep_graph}: {', '.join(obj_details)}")
            
            # If no tasks are explicitly listed, create a default task entry
            if not graph.tasks:
                task_data.append({
                    "TaskGraph": graph.graph_id,
                    "Task": f"{graph.graph_id}_main",
                    "Device": graph.device,
                    "Allocations": num_allocs,
                    "Deallocations": num_deallocs,
                    "PersistedObjects": num_persisted,
                    "TotalMemoryAllocated (MB)": f"{mem_allocated/(1024*1024):.2f}",
                    "TotalMemoryTransferred (MB)": f"{mem_transferred/(1024*1024):.2f}",
                    "Dependencies": "\n".join(dep_details) if dep_details else "None",
                    "NumOperations": len(graph.operations)
                })
            else:
                # Create an entry for each task in the graph
                for task_name in graph.tasks:
                    task_data.append({
                        "TaskGraph": graph.graph_id,
                        "Task": task_name,
                        "Device": graph.device,
                        "Allocations": num_allocs,
                        "Deallocations": num_deallocs,
                        "PersistedObjects": num_persisted,
                        "TotalMemoryAllocated (MB)": f"{mem_allocated/(1024*1024):.2f}",
                        "TotalMemoryTransferred (MB)": f"{mem_transferred/(1024*1024):.2f}",
                        "Dependencies": "\n".join(dep_details) if dep_details else "None",
                        "NumOperations": len(graph.operations)
                    })
        
        # Create DataFrame and ensure it's not empty
        df = pd.DataFrame(task_data)
        if df.empty:
            # Create a dummy row if no data
            df = pd.DataFrame([{
                "TaskGraph": "No Data",
                "Task": "No Tasks Found",
                "Device": "N/A",
                "Allocations": 0,
                "Deallocations": 0,
                "PersistedObjects": 0,
                "TotalMemoryAllocated (MB)": "0.00",
                "TotalMemoryTransferred (MB)": "0.00",
                "Dependencies": "None",
                "NumOperations": 0
            }])
        
        return df
    
    def get_memory_usage_chart(self) -> go.Figure:
        """Generate a chart showing memory usage over time"""
        # Track memory allocations and deallocations over time
        memory_events = []
        
        for i, graph in enumerate(self.task_graphs):
            base_index = sum(len(g.operations) for g in self.task_graphs[:i])
            
            for j, op in enumerate(graph.operations):
                global_index = base_index + j
                
                if op.operation == "ALLOC":
                    for obj_ref in op.objects:
                        obj_hash = self._extract_hash(obj_ref)
                        memory_events.append({
                            "GlobalIndex": global_index,
                            "TaskGraph": graph.graph_id,
                            "Operation": "Allocation",
                            "Size": op.size,
                            "Object": obj_hash
                        })
                        
                elif op.operation == "DEALLOC":
                    for obj_ref in op.objects:
                        obj_hash = self._extract_hash(obj_ref)
                        # Only count as deallocation if actually freed
                        if "Freed" in op.status:
                            memory_events.append({
                                "GlobalIndex": global_index,
                                "TaskGraph": graph.graph_id,
                                "Operation": "Deallocation",
                                "Size": -1 * self._get_object_size(obj_hash),  # Negative size for deallocation
                                "Object": obj_hash
                            })
        
        # Convert to DataFrame
        df = pd.DataFrame(memory_events)
        if df.empty:
            fig = go.Figure()
            fig.update_layout(
                title="Memory Usage Over Time (No Data)",
                xaxis_title="Operation Sequence",
                yaxis_title="Memory Usage (bytes)",
            )
            return fig
            
        # Calculate cumulative memory usage
        df = df.sort_values("GlobalIndex")
        df["CumulativeMemory"] = df["Size"].cumsum()
        
        # Create the chart
        fig = go.Figure()
        
        # Add memory usage line
        fig.add_trace(go.Scatter(
            x=df["GlobalIndex"],
            y=df["CumulativeMemory"],
            mode="lines",
            name="Memory Usage",
            line=dict(color="rgba(52, 152, 219, 1)", width=3),
            fill="tozeroy",
            fillcolor="rgba(52, 152, 219, 0.2)"
        ))
        
        # Add markers for allocation events
        allocs = df[df["Operation"] == "Allocation"]
        if not allocs.empty:
            fig.add_trace(go.Scatter(
                x=allocs["GlobalIndex"],
                y=allocs["CumulativeMemory"],
                mode="markers",
                marker=dict(color="green", size=8, symbol="circle"),
                name="Allocations",
                text=allocs.apply(
                    lambda row: f"Allocated {row['Size']:,} bytes<br>Object: {row['Object']}<br>In {row['TaskGraph']}",
                    axis=1
                ),
                hoverinfo="text"
            ))
        
        # Add markers for deallocation events
        deallocs = df[df["Operation"] == "Deallocation"]
        if not deallocs.empty:
            fig.add_trace(go.Scatter(
                x=deallocs["GlobalIndex"],
                y=deallocs["CumulativeMemory"],
                mode="markers",
                marker=dict(color="red", size=8, symbol="x"),
                name="Deallocations",
                text=deallocs.apply(
                    lambda row: f"Deallocated {abs(row['Size']):,} bytes<br>Object: {row['Object']}<br>In {row['TaskGraph']}",
                    axis=1
                ),
                hoverinfo="text"
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': "Memory Usage Over Time",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 18}
            },
            xaxis_title="Operation Sequence",
            yaxis_title="Memory Usage (bytes)",
            hovermode="closest",
            height=400,
            width=600,
            template="plotly_white",
        )
        
        return fig
    
    def _get_object_size(self, obj_hash: str) -> int:
        """Helper to get object size from hash"""
        if obj_hash in self.memory_objects:
            return self.memory_objects[obj_hash].size
        return 0
    
    def get_object_persistence_chart(self) -> go.Figure:
        """Create a chart showing object persistence patterns"""
        # Group objects by their status
        status_counts = defaultdict(int)
        size_by_status = defaultdict(int)
        
        for obj_hash, obj in self.memory_objects.items():
            status = obj.current_status
            status_counts[status] += 1
            size_by_status[status] += obj.size
        
        # Convert to DataFrames
        count_df = pd.DataFrame({
            "Status": list(status_counts.keys()),
            "Count": list(status_counts.values())
        })
        
        if count_df.empty:
            fig = go.Figure()
            fig.update_layout(
                title="Object Persistence Analysis (No Data)",
                height=400, 
                width=600
            )
            return fig
        
        # Create pie chart
        fig = go.Figure()
        
        fig.add_trace(go.Pie(
            labels=count_df["Status"],
            values=count_df["Count"],
            hole=0.4,
            textinfo="label+percent",
            hoverinfo="label+value+percent"
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': "Object Persistence Analysis",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 18}
            },
            height=400,
            width=600,
            template="plotly_white",
            showlegend=True
        )
        
        return fig
    
    def get_bytecode_distribution_chart(self) -> go.Figure:
        """Create a chart showing bytecode operation distribution"""
        # Count bytecode operations by type
        op_counts = defaultdict(int)
        for bc in self.bytecode_details:
            op_counts[bc["Operation"]] += 1
        
        # Convert to DataFrame
        df = pd.DataFrame({
            "Operation": list(op_counts.keys()),
            "Count": list(op_counts.values())
        }).sort_values("Count", ascending=False)
        
        if df.empty:
            fig = go.Figure()
            fig.update_layout(
                title="Bytecode Operation Distribution (No Data)",
                height=400, 
                width=600
            )
            return fig
        
        # Create pie chart
        fig = go.Figure()
        
        fig.add_trace(go.Pie(
            labels=df["Operation"],
            values=df["Count"],
            hole=0.4,
            textinfo="label+percent",
            hoverinfo="label+value+percent"
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': "Bytecode Operation Distribution",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 18}
            },
            height=400,
            width=600,
            template="plotly_white",
            showlegend=False
        )
        
        return fig

# Main Streamlit application
def main():
    # Apply custom CSS for dark theme
    st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #1e2130;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .dashboard-title {
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # App header with logo
    st.markdown("""
    <div class="dashboard-title">
        <h1>🌪️ TornadoVM Bytecode Visualizer</h1>
        <p>Analyze and visualize TornadoVM bytecode execution patterns and memory operations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for navigation and file upload
    with st.sidebar:
        st.header("Upload Data")
        uploaded_file = st.file_uploader("Upload TornadoVM bytecode log", type=["txt", "log"])
        
        if uploaded_file:
            st.success(f"File uploaded: {uploaded_file.name}")
            page = st.radio(
                "Select View:",
                ["Basic Overview", "Task Graphs", "Memory Analysis", "Bytecode Details"],
                index=0
            )
        else:
            st.info("Please upload a TornadoVM bytecode log to begin")
            page = "Welcome"
    
    # Show welcome screen when no file is uploaded
    if not uploaded_file:
        st.header("Welcome to TornadoVM Bytecode Visualizer")
        
        st.markdown("""
        This tool helps you analyze TornadoVM bytecode execution logs to understand:
        
        - **Task Graph Structure**: Visualize how task graphs depend on each other
        - **Memory Operations**: Track allocations, transfers, and deallocations
        - **Data Dependencies**: See how objects flow between different task graphs
        - **Performance Insights**: Identify potential memory bottlenecks
        
        ### Getting Started
        
        1. Upload a TornadoVM bytecode log file using the sidebar
        2. Explore different visualizations using the navigation options
        3. Gain insights into your TornadoVM application's behavior
        
        ### Sample Visualization
        """)
        
        # Show example image
        st.image("https://via.placeholder.com/800x400.png?text=TornadoVM+Task+Graph+Visualization", 
                caption="Example task graph visualization")
        return
    
    # Process uploaded file
    try:
        log_content = uploaded_file.read().decode("utf-8")
        visualizer = TornadoVisualizer()
        visualizer.parse_log(log_content)
        
        # Basic metrics
        num_task_graphs = len(visualizer.task_graphs)
        total_objects = len(visualizer.memory_objects)
        total_bytecodes = len(visualizer.bytecode_details)
        # Calculate total tasks by counting LAUNCH operations
        total_tasks = sum(1 for bc in visualizer.bytecode_details 
                         if bc["Operation"] == "LAUNCH" and bc["TaskName"])
        # If no LAUNCH operations found, count task graph entries
        if total_tasks == 0:
            total_tasks = sum(len(graph.tasks) for graph in visualizer.task_graphs)

        # Calculate memory metrics
        total_allocated = sum(obj.size for obj in visualizer.memory_objects.values())
        total_persisted = sum(obj.size for obj in visualizer.memory_objects.values() 
                             if "Persisted" in obj.current_status)
        
        # Get summary dataframe with enhanced task details
        summary_df = visualizer.generate_task_summary()
        
        # Different pages based on sidebar selection
        if page == "Basic Overview":
            st.header("Overview Dashboard")
            
            # Display key metrics in a row
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Task Graphs", num_task_graphs)
            with col2:
                st.metric("Total Tasks", total_tasks)
            with col3:
                st.metric("Total Objects", total_objects)
            with col4:
                st.metric("Memory (MB)", f"{total_allocated/(1024*1024):.2f}")
            with col5:
                st.metric("Total Bytecodes", total_bytecodes)
            
            # Display summary table
            st.subheader("Task Graph Summary")
            # Configure the dataframe display
            st.markdown("""
            <style>
            .task-summary {
                white-space: pre-wrap !important;
                font-family: monospace !important;
            }
            .stDataFrame {
                width: 100% !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Display with custom formatting
            st.dataframe(
                summary_df,
                column_config={
                    "Dependencies": st.column_config.TextColumn(
                        "Dependencies",
                        help="Object dependencies between task graphs",
                        max_chars=1000,
                        width="large"
                    ),
                    "Task": st.column_config.TextColumn(
                        "Task",
                        help="Individual task name",
                        width="medium"
                    ),
                    "TaskGraph": st.column_config.TextColumn(
                        "TaskGraph",
                        help="Task graph identifier",
                        width="medium"
                    ),
                    "Device": st.column_config.TextColumn(
                        "Device",
                        help="Execution device",
                        width="medium"
                    ),
                    "NumOperations": st.column_config.NumberColumn(
                        "Operations",
                        help="Number of operations in the task graph",
                        format="%d"
                    ),
                    "TotalMemoryAllocated (MB)": st.column_config.NumberColumn(
                        "Memory Allocated (MB)",
                        help="Total memory allocated in MB",
                        format="%.2f"
                    ),
                    "TotalMemoryTransferred (MB)": st.column_config.NumberColumn(
                        "Memory Transferred (MB)",
                        help="Total memory transferred in MB",
                        format="%.2f"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Simple dependency graph
            st.subheader("Task Graph Dependencies")
            simple_graph = visualizer.visualize_simple_dependency_graph()
            if simple_graph:
                st.pyplot(simple_graph)
            else:
                st.info("No task graph dependencies to display")
                
            # Basic charts in two columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Memory Usage")
                mem_chart = visualizer.get_memory_usage_chart()
                st.plotly_chart(mem_chart, use_container_width=True)
                
            with col2:
                st.subheader("Bytecode Distribution")
                bc_chart = visualizer.get_bytecode_distribution_chart()
                st.plotly_chart(bc_chart, use_container_width=True)
        
        elif page == "Task Graphs":
            st.header("Task Graph Analysis")
            
            # Detailed dependency visualization
            st.subheader("Task Graph Dependencies")
            detailed_graph = visualizer.visualize_dependency_graph_detailed()
            if detailed_graph:
                st.pyplot(detailed_graph)
            else:
                st.info("No task graph dependencies to display")
            
            # Task details
            st.subheader("Task Details")
            
            # Select a specific task graph
            if visualizer.task_graphs:
                selected_graph = st.selectbox(
                    "Select Task Graph:",
                    options=[graph.graph_id for graph in visualizer.task_graphs]
                )
                
                # Show selected graph details
                selected = next((g for g in visualizer.task_graphs if g.graph_id == selected_graph), None)
                if selected:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Device:** {selected.device}")
                        st.markdown(f"**Operations:** {len(selected.operations)}")
                        
                        # Show tasks
                        st.markdown("**Tasks:**")
                        for task in selected.tasks:
                            st.markdown(f"- {task}")
                    
                    with col2:
                        # Show dependencies
                        st.markdown("**Dependencies:**")
                        if selected.dependencies:
                            for dep, objs in selected.dependencies.items():
                                st.markdown(f"- {dep} ({len(objs)} objects)")
                        else:
                            st.markdown("No dependencies")
                    
                    # Show bytecodes
                    st.subheader("Bytecode Operations")
                    bytecode_df = pd.DataFrame([
                        {"Operation": op.operation, 
                         "Objects": ", ".join(op.objects), 
                         "Task": op.task_name,
                         "Size": op.size if hasattr(op, "size") and op.size else 0,
                         "Status": op.status if hasattr(op, "status") and op.status else ""}
                        for op in selected.operations
                    ])
                    st.dataframe(bytecode_df)
            else:
                st.info("No task graphs found in the log file")
        
        elif page == "Memory Analysis":
            st.header("Memory Analysis")
            
            # Memory timeline
            st.subheader("Memory Operations Timeline")
            
            # Simplify to a basic chart if the interactive one fails
            try:
                timeline_fig = visualizer.visualize_memory_timeline_interactive()
                st.plotly_chart(timeline_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating memory timeline: {e}")
                st.info("Showing simplified memory usage chart instead")
                mem_chart = visualizer.get_memory_usage_chart()
                st.plotly_chart(mem_chart, use_container_width=True)
            
            # Object details
            st.subheader("Object Analysis")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Object selection dropdown
                object_options = []
                for obj_id, obj in visualizer.memory_objects.items():
                    short_type = obj.object_type.split('.')[-1]
                    object_options.append((f"{short_type}@{obj_id[:8]}", obj_id))
                
                if object_options:
                    selected_object = st.selectbox(
                        "Select Object:", 
                        options=[obj_id for _, obj_id in object_options],
                        format_func=lambda x: next((name for name, id in object_options if id == x), x)
                    )
                    
                    # Show object details
                    if selected_object in visualizer.memory_objects:
                        obj = visualizer.memory_objects[selected_object]
                        st.markdown(f"**Type:** {obj.object_type}")
                        st.markdown(f"**Size:** {obj.size:,} bytes")
                        st.markdown(f"**Status:** {obj.current_status}")
                        st.markdown(f"**Allocated in:** {obj.allocated_in_graph}")
                else:
                    st.info("No objects found in the log file")
                    selected_object = None
            
            with col2:
                # Object flow visualization
                if object_options and selected_object:
                    try:
                        flow_fig = visualizer.visualize_object_flow(selected_object)
                        st.plotly_chart(flow_fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating object flow: {e}")
            
            # Memory statistics charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Memory Usage Over Time")
                mem_chart = visualizer.get_memory_usage_chart()
                st.plotly_chart(mem_chart, use_container_width=True)
            
            with col2:
                st.subheader("Object Persistence")
                persistence_chart = visualizer.get_object_persistence_chart()
                st.plotly_chart(persistence_chart, use_container_width=True)
        
        elif page == "Bytecode Details":
            st.header("Bytecode Analysis")
            
            # Bytecode distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Bytecode Distribution")
                bc_chart = visualizer.get_bytecode_distribution_chart()
                st.plotly_chart(bc_chart, use_container_width=True)
            
            with col2:
                # Calculate operation counts by task graph
                if visualizer.bytecode_details:
                    bc_by_graph = defaultdict(lambda: defaultdict(int))
                    for bc in visualizer.bytecode_details:
                        bc_by_graph[bc["TaskGraph"]][bc["Operation"]] += 1
                    
                    # Create heatmap data
                    graph_ops = []
                    for graph, ops in bc_by_graph.items():
                        for op, count in ops.items():
                            graph_ops.append({"TaskGraph": graph, "Operation": op, "Count": count})
                    
                    graph_ops_df = pd.DataFrame(graph_ops)
                    
                    if not graph_ops_df.empty:
                        st.subheader("Operations by Task Graph")
                        # Create a simple table instead of heatmap
                        pivot_df = graph_ops_df.pivot_table(
                            index="TaskGraph", 
                            columns="Operation", 
                            values="Count",
                            aggfunc='sum',
                            fill_value=0
                        )
                        st.dataframe(pivot_df)
            
            # Bytecode listing with filters
            st.subheader("Bytecode Listing")
            
            if visualizer.bytecode_details:
                # Filters
                col1, col2, col3 = st.columns(3)
                with col1:
                    graph_filter = st.multiselect(
                        "Filter by Task Graph",
                        options=sorted(set(bc["TaskGraph"] for bc in visualizer.bytecode_details))
                    )
                with col2:
                    op_filter = st.multiselect(
                        "Filter by Operation",
                        options=sorted(set(bc["Operation"] for bc in visualizer.bytecode_details))
                    )
                with col3:
                    search_term = st.text_input("Search Objects")
                
                # Apply filters
                filtered_bc = visualizer.bytecode_details
                if graph_filter:
                    filtered_bc = [bc for bc in filtered_bc if bc["TaskGraph"] in graph_filter]
                if op_filter:
                    filtered_bc = [bc for bc in filtered_bc if bc["Operation"] in op_filter]
                if search_term:
                    filtered_bc = [bc for bc in filtered_bc if search_term.lower() in bc.get("Objects", "").lower()]
                
                # Display filtered data
                bc_df = pd.DataFrame(filtered_bc)
                if not bc_df.empty:
                    st.dataframe(bc_df)
                else:
                    st.info("No bytecodes match the selected filters")
            else:
                st.info("No bytecode details found in the log file")
    
    except Exception as e:
        st.error(f"Error processing log file: {str(e)}")
        st.info("Please check that the file contains valid TornadoVM bytecode logs")

if __name__ == "__main__":
    main()