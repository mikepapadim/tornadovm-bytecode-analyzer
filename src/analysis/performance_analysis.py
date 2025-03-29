import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from datetime import datetime
import sqlite3

@dataclass
class PerformanceMetrics:
    """Stores performance metrics for a code block"""
    block_type: str
    metrics: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class PerformanceAnalyzer:
    """Analyzes performance characteristics of PTX and OpenCL code"""
    
    def __init__(self):
        self.db_conn = self._init_database()
        
    def _init_database(self) -> sqlite3.Connection:
        """Initialize SQLite database for metrics tracking"""
        conn = sqlite3.connect(':memory:')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS performance_metrics
                    (timestamp TEXT, block_type TEXT, metric_type TEXT, 
                     metric_name TEXT, value REAL)''')
        conn.commit()
        return conn
    
    def analyze_ptx_code(self, code: str) -> PerformanceMetrics:
        """Analyze PTX code for performance characteristics"""
        metrics = PerformanceMetrics(block_type='ptx')
        
        # Analyze instruction mix
        instruction_mix = self._analyze_instruction_mix(code)
        metrics.metrics['instruction_mix'] = instruction_mix
        
        # Analyze register usage
        register_usage = self._analyze_register_usage(code)
        metrics.metrics['register_usage'] = register_usage
        
        # Analyze memory operations
        memory_ops = self._analyze_memory_operations(code)
        metrics.metrics['memory_operations'] = memory_ops
        
        # Calculate derived metrics
        metrics.metrics['arithmetic_intensity'] = self._calculate_arithmetic_intensity(
            instruction_mix, memory_ops
        )
        
        # Store metrics in database
        self._store_metrics(metrics)
        
        return metrics
    
    def analyze_opencl_code(self, code: str) -> PerformanceMetrics:
        """Analyze OpenCL code for performance characteristics"""
        metrics = PerformanceMetrics(block_type='opencl')
        
        # Analyze memory access patterns
        memory_patterns = self._analyze_memory_patterns(code)
        metrics.metrics['memory_patterns'] = memory_patterns
        
        # Analyze work group utilization
        work_group = self._analyze_work_group_usage(code)
        metrics.metrics['work_group'] = work_group
        
        # Analyze vectorization
        vectorization = self._analyze_vectorization(code)
        metrics.metrics['vectorization'] = vectorization
        
        # Analyze thread divergence
        divergence = self._analyze_thread_divergence(code)
        metrics.metrics['thread_divergence'] = divergence
        
        # Store metrics in database
        self._store_metrics(metrics)
        
        return metrics
    
    def _analyze_instruction_mix(self, code: str) -> Dict[str, int]:
        """Analyze PTX instruction mix"""
        instruction_types = {
            'arithmetic': r'\b(add|sub|mul|div|mad)\.',
            'memory': r'\b(ld|st|atom)\.',
            'control': r'\b(bra|ret|call)\b',
            'conversion': r'\bcvt\.',
            'special': r'\b(sin|cos|sqrt|rcp)\.'
        }
        
        counts = {category: 0 for category in instruction_types}
        lines = code.split('\n')
        
        for line in lines:
            for category, pattern in instruction_types.items():
                if re.search(pattern, line):
                    counts[category] += 1
        
        return counts
    
    def _analyze_register_usage(self, code: str) -> Dict[str, List[str]]:
        """Analyze PTX register usage"""
        usage = {
            'declarations': [],
            'reads': [],
            'writes': [],
            'reuse': []
        }
        
        reg_pattern = r'%r\d+'
        lines = code.split('\n')
        reg_defs = {}  # Track register definitions
        
        for i, line in enumerate(lines):
            # Track register declarations
            if '.reg' in line:
                usage['declarations'].append(line.strip())
            
            # Track register reads and writes
            regs = re.findall(reg_pattern, line)
            if regs:
                for reg in regs:
                    if '=' in line or 'st.' in line:
                        usage['writes'].append((reg, i, line.strip()))
                        reg_defs[reg] = i
                    else:
                        usage['reads'].append((reg, i, line.strip()))
                        # Check for register reuse
                        if reg in reg_defs and i - reg_defs[reg] < 5:
                            usage['reuse'].append((reg, reg_defs[reg], i, line.strip()))
        
        return usage
    
    def _analyze_memory_operations(self, code: str) -> Dict[str, List[Dict]]:
        """Analyze PTX memory operations"""
        operations = {
            'global_loads': [],
            'global_stores': [],
            'shared_memory': [],
            'atomic': []
        }
        
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if 'ld.global' in line:
                operations['global_loads'].append({
                    'line': i,
                    'code': line.strip(),
                    'coalesced': 'mad.' in line or 'add.' in line
                })
            elif 'st.global' in line:
                operations['global_stores'].append({
                    'line': i,
                    'code': line.strip(),
                    'coalesced': 'mad.' in line or 'add.' in line
                })
            elif '.shared' in line:
                operations['shared_memory'].append({
                    'line': i,
                    'code': line.strip()
                })
            elif 'atom.' in line:
                operations['atomic'].append({
                    'line': i,
                    'code': line.strip()
                })
        
        return operations
    
    def _analyze_memory_patterns(self, code: str) -> Dict[str, List[Dict]]:
        """Analyze OpenCL memory access patterns"""
        patterns = {
            'global_reads': [],
            'global_writes': [],
            'local_memory': [],
            'private_memory': [],
            'atomic_operations': []
        }
        
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if '__global' in line:
                if '*' in line and '=' in line:
                    patterns['global_writes'].append({
                        'line': i,
                        'code': line.strip(),
                        'vectorized': any(f'float{n}' in line or f'int{n}' in line 
                                        for n in [2,4,8,16])
                    })
                else:
                    patterns['global_reads'].append({
                        'line': i,
                        'code': line.strip(),
                        'vectorized': any(f'float{n}' in line or f'int{n}' in line 
                                        for n in [2,4,8,16])
                    })
            elif '__local' in line:
                patterns['local_memory'].append({
                    'line': i,
                    'code': line.strip()
                })
            elif 'private' in line:
                patterns['private_memory'].append({
                    'line': i,
                    'code': line.strip()
                })
            elif 'atomic_' in line:
                patterns['atomic_operations'].append({
                    'line': i,
                    'code': line.strip()
                })
        
        return patterns
    
    def _analyze_work_group_usage(self, code: str) -> Dict[str, List[Dict]]:
        """Analyze OpenCL work group usage"""
        usage = {
            'local_size': [],
            'barriers': [],
            'work_item_ops': []
        }
        
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if 'get_local_size' in line:
                usage['local_size'].append({
                    'line': i,
                    'code': line.strip(),
                    'dimension': re.search(r'get_local_size\((\d+)\)', line).group(1)
                    if re.search(r'get_local_size\((\d+)\)', line) else None
                })
            elif 'barrier(' in line:
                usage['barriers'].append({
                    'line': i,
                    'code': line.strip(),
                    'type': 'local' if 'CLK_LOCAL_MEM_FENCE' in line else 'global'
                })
            elif 'get_local_id' in line:
                usage['work_item_ops'].append({
                    'line': i,
                    'code': line.strip(),
                    'dimension': re.search(r'get_local_id\((\d+)\)', line).group(1)
                    if re.search(r'get_local_id\((\d+)\)', line) else None
                })
        
        return usage
    
    def _analyze_vectorization(self, code: str) -> Dict[str, List[Dict]]:
        """Analyze OpenCL vectorization opportunities"""
        vectorization = {
            'vector_types': [],
            'vector_ops': [],
            'scalar_ops': []
        }
        
        vector_types = [f'{t}{n}' for t in ['float', 'int', 'uint'] 
                       for n in [2,4,8,16]]
        
        lines = code.split('\n')
        for i, line in enumerate(lines):
            # Check for vector type declarations
            for vtype in vector_types:
                if vtype in line:
                    vectorization['vector_types'].append({
                        'line': i,
                        'code': line.strip(),
                        'type': vtype
                    })
            
            # Check for vector operations
            if any(op in line for op in ['.s0123', '.even', '.odd', '.hi', '.lo']):
                vectorization['vector_ops'].append({
                    'line': i,
                    'code': line.strip()
                })
            # Check for potential vectorization opportunities
            elif re.search(r'float\s+\w+\s*=|int\s+\w+\s*=', line):
                vectorization['scalar_ops'].append({
                    'line': i,
                    'code': line.strip()
                })
        
        return vectorization
    
    def _analyze_thread_divergence(self, code: str) -> Dict[str, List[Dict]]:
        """Analyze potential thread divergence in OpenCL code"""
        divergence = {
            'conditional_branches': [],
            'loop_divergence': [],
            'atomic_operations': []
        }
        
        lines = code.split('\n')
        for i, line in enumerate(lines):
            # Check for conditionals based on thread ID
            if 'if' in line and ('get_global_id' in line or 'get_local_id' in line):
                divergence['conditional_branches'].append({
                    'line': i,
                    'code': line.strip(),
                    'severity': 'high' if 'get_global_id' in line else 'medium'
                })
            
            # Check for loops with thread-dependent bounds
            if ('for' in line or 'while' in line) and \
               ('get_global_id' in line or 'get_local_id' in line):
                divergence['loop_divergence'].append({
                    'line': i,
                    'code': line.strip(),
                    'severity': 'high'
                })
            
            # Check for atomic operations
            if 'atomic_' in line:
                divergence['atomic_operations'].append({
                    'line': i,
                    'code': line.strip(),
                    'severity': 'medium'
                })
        
        return divergence
    
    def _calculate_arithmetic_intensity(self, instruction_mix: Dict[str, int], 
                                     memory_ops: Dict[str, List[Dict]]) -> float:
        """Calculate arithmetic intensity (ops per byte)"""
        arithmetic_ops = instruction_mix.get('arithmetic', 0)
        memory_accesses = (len(memory_ops.get('global_loads', [])) + 
                         len(memory_ops.get('global_stores', [])))
        
        return arithmetic_ops / memory_accesses if memory_accesses > 0 else 0
    
    def _store_metrics(self, metrics: PerformanceMetrics) -> None:
        """Store metrics in SQLite database"""
        cursor = self.db_conn.cursor()
        
        def flatten_metrics(prefix: str, metric_dict: Dict) -> List[tuple]:
            flattened = []
            for key, value in metric_dict.items():
                if isinstance(value, dict):
                    flattened.extend(flatten_metrics(f"{prefix}_{key}", value))
                elif isinstance(value, (int, float)):
                    flattened.append((
                        metrics.timestamp,
                        metrics.block_type,
                        prefix,
                        key,
                        float(value)
                    ))
                elif isinstance(value, list):
                    flattened.append((
                        metrics.timestamp,
                        metrics.block_type,
                        prefix,
                        key,
                        float(len(value))
                    ))
            return flattened
        
        # Flatten and store all metrics
        flattened_metrics = flatten_metrics("", metrics.metrics)
        cursor.executemany(
            "INSERT INTO performance_metrics VALUES (?, ?, ?, ?, ?)",
            flattened_metrics
        )
        self.db_conn.commit()
    
    def get_metrics_history(self, block_type: str = None, 
                          metric_type: str = None) -> List[Dict]:
        """Retrieve metrics history from database"""
        cursor = self.db_conn.cursor()
        
        query = "SELECT * FROM performance_metrics"
        params = []
        
        if block_type or metric_type:
            conditions = []
            if block_type:
                conditions.append("block_type = ?")
                params.append(block_type)
            if metric_type:
                conditions.append("metric_type = ?")
                params.append(metric_type)
            query += " WHERE " + " AND ".join(conditions)
        
        cursor.execute(query, params)
        
        return [
            {
                'timestamp': row[0],
                'block_type': row[1],
                'metric_type': row[2],
                'metric_name': row[3],
                'value': row[4]
            }
            for row in cursor.fetchall()
        ] 