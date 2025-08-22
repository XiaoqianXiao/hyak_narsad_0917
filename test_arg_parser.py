#!/usr/bin/env python3
"""
Test script for argument parser validation
Author: Xiaoqian Xiao (xiao.xiaoqian.320@gmail.com)
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

# Mock the nipype imports to test argument parsing
class MockWorkflow:
    def __init__(self, name):
        self.name = name

class MockNode:
    pass

class MockMapNode:
    pass

# Mock the imports before importing the main script
import types
mock_nipype = types.ModuleType('nipype.pipeline.engine')
mock_nipype.Workflow = MockWorkflow
mock_nipype.Node = MockNode
mock_nipype.MapNode = MockMapNode

# Create the nipype module structure
sys.modules['nipype'] = types.ModuleType('nipype')
sys.modules['nipype.pipeline'] = types.ModuleType('nipype.pipeline')
sys.modules['nipype.pipeline.engine'] = mock_nipype

# Now import the main script
try:
    exec(open('run_group_voxelWise.py').read().split('if __name__')[0])
    print("✅ Argument parser setup is valid")
    print("✅ All imports and function definitions parsed successfully")
    
    # Test argument parsing
    test_args = [
        '--task', 'phase2',
        '--contrast', '1',
        '--base-dir', '/test/path',
        '--analysis-type', 'randomise'
    ]
    
    # Temporarily replace sys.argv for testing
    original_argv = sys.argv
    sys.argv = ['test_script.py'] + test_args
    
    try:
        # This should work without errors
        args = parser.parse_args()
        print(f"✅ Argument parsing successful:")
        print(f"   Task: {args.task}")
        print(f"   Contrast: {args.contrast}")
        print(f"   Base dir: {args.base_dir}")
        print(f"   Analysis type: {args.analysis_type}")
    except Exception as e:
        print(f"❌ Argument parsing failed: {e}")
    
    # Restore original argv
    sys.argv = original_argv
    
except Exception as e:
    print(f"❌ Script parsing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
