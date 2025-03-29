import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple

@dataclass
class SemanticPattern:
    """Represents a semantic code pattern with its mappings"""
    pattern_type: str
    java_pattern: str
    ptx_pattern: str
    opencl_pattern: str
    description: str

class SemanticAnalyzer:
    """Analyzes semantic patterns in code transitions"""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
        
    def _initialize_patterns(self) -> Dict[str, SemanticPattern]:
        """Initialize known semantic patterns for code transitions"""
        return {
            'parallel_loop': SemanticPattern(
                pattern_type='parallel_loop',
                java_pattern=r'@Parallel\s+for\s*\([^)]*\)',
                ptx_pattern=r'mov\.u32\s+.*%ntid|mad\.lo\.s32',
                opencl_pattern=r'get_global_id|get_local_id',
                description='Parallel loop transformation'
            ),
            'array_access': SemanticPattern(
                pattern_type='array_access',
                java_pattern=r'\.get\s*\(\s*(\w+)\s*\)|\.set\s*\(\s*(\w+)',
                ptx_pattern=r'ld\.global|st\.global',
                opencl_pattern=r'__global\s+\w+\s*\*',
                description='Array access pattern'
            ),
            'math_operations': SemanticPattern(
                pattern_type='math_operations',
                java_pattern=r'TornadoMath\.(sin|cos|sqrt)',
                ptx_pattern=r'(sin|cos|sqrt)\.approx',
                opencl_pattern=r'native_(sin|cos|sqrt)',
                description='Mathematical operation'
            ),
            'reduction': SemanticPattern(
                pattern_type='reduction',
                java_pattern=r'@Reduce\s*\([^)]*\)',
                ptx_pattern=r'(add|max|min)\.red\.[us]32',
                opencl_pattern=r'atomic_(add|max|min)',
                description='Reduction operation'
            ),
            'vector_operation': SemanticPattern(
                pattern_type='vector_operation',
                java_pattern=r'VectorFloat\d+|VectorDouble\d+',
                ptx_pattern=r'\.v[2-4]\.(f32|f64)',
                opencl_pattern=r'(float|double)\d+',
                description='Vector operation'
            ),
            'barrier_sync': SemanticPattern(
                pattern_type='barrier_sync',
                java_pattern=r'@Parallel\s+barrier\s*\(\s*\)',
                ptx_pattern=r'bar\.sync',
                opencl_pattern=r'barrier\s*\(',
                description='Barrier synchronization'
            ),
            'memory_fence': SemanticPattern(
                pattern_type='memory_fence',
                java_pattern=r'@Parallel\s+fence\s*\(\s*\)',
                ptx_pattern=r'membar\.(gl|sys)',
                opencl_pattern=r'mem_fence\s*\(',
                description='Memory fence operation'
            )
        }
    
    def find_patterns(self, code: str, language: str) -> List[Dict]:
        """Find semantic patterns in code for a specific language"""
        matches = []
        
        for pattern_name, pattern in self.patterns.items():
            pattern_regex = getattr(pattern, f'{language}_pattern')
            if not pattern_regex:
                continue
                
            for match in re.finditer(pattern_regex, code):
                matches.append({
                    'pattern_type': pattern_name,
                    'start': match.start(),
                    'end': match.end(),
                    'text': match.group(0),
                    'line': code.count('\n', 0, match.start()) + 1,
                    'description': pattern.description
                })
        
        return sorted(matches, key=lambda x: x['start'])
    
    def analyze_code_mapping(self, java_code: str, target_code: str, target_type: str) -> List[Dict]:
        """Analyze semantic mappings between Java and target code"""
        java_patterns = self.find_patterns(java_code, 'java')
        target_patterns = self.find_patterns(target_code, target_type)
        
        mappings = []
        for java_pat in java_patterns:
            # Find potential matches in target code
            matching_targets = [
                target for target in target_patterns
                if target['pattern_type'] == java_pat['pattern_type']
            ]
            
            if matching_targets:
                mappings.append({
                    'java_pattern': java_pat,
                    'target_patterns': matching_targets,
                    'pattern_type': java_pat['pattern_type'],
                    'description': self.patterns[java_pat['pattern_type']].description
                })
        
        return mappings
    
    def get_pattern_description(self, pattern_type: str) -> str:
        """Get detailed description of a pattern type"""
        if pattern_type in self.patterns:
            return self.patterns[pattern_type].description
        return "Unknown pattern type" 