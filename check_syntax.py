#!/usr/bin/env python3
"""
Syntax check for all Python files in the project.
"""

import ast
import sys
import os

def check_syntax(filepath):
    """Check syntax of a single Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, e
    except Exception as e:
        return False, e

def main():
    """Check syntax of all Python files in the project."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    python_files = []
    
    for root, dirs, files in os.walk(project_root):
        # Skip __pycache__ and similar directories
        dirs[:] = [d for d in dirs if not d.startswith('_') and not d.startswith('.')]
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    if not python_files:
        print("No Python files found.")
        return 1
    
    all_passed = True
    for filepath in python_files:
        rel_path = os.path.relpath(filepath, project_root)
        passed, error = check_syntax(filepath)
        if passed:
            print(f"✓ {rel_path}")
        else:
            print(f"✗ {rel_path}")
            print(f"  Error: {error}")
            all_passed = False
    
    if all_passed:
        print("\nAll Python files have valid syntax.")
        return 0
    else:
        print("\nSyntax errors found.")
        return 1

if __name__ == "__main__":
    sys.exit(main())