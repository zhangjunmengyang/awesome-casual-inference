#!/usr/bin/env python3
"""
Review Part 6 Marketing notebooks for completeness and quality
"""
import json
import re
from pathlib import Path

def extract_code_cells(notebook_path):
    """Extract all code cells from a notebook"""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    code_cells = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            code_cells.append(source)

    return code_cells

def check_for_issues(code_cells):
    """Check for common issues in code cells"""
    issues = {
        'empty_functions': [],
        'unimplemented_todos': [],
        'error_prone': []
    }

    for i, cell in enumerate(code_cells):
        # Check for empty function definitions
        if re.search(r'def\s+\w+.*:\s*pass\s*$', cell, re.MULTILINE):
            issues['empty_functions'].append((i, cell[:100]))

        # Check for actual TODOs (not in comments or strings)
        if 'TODO' in cell and '# TODO' not in cell and '"TODO"' not in cell and "'TODO'" not in cell:
            issues['unimplemented_todos'].append((i, cell[:100]))

        # Check for potential errors
        if re.search(r'return\s+None\s*$', cell, re.MULTILINE):
            issues['error_prone'].append((i, 'Returns None'))

    return issues

def main():
    """Main review function"""
    notebooks = [
        'part6_1_marketing_attribution.ipynb',
        'part6_2_coupon_optimization.ipynb',
        'part6_3_user_targeting.ipynb',
        'part6_4_budget_allocation.ipynb'
    ]

    for nb_name in notebooks:
        print(f'\n{"="*60}')
        print(f'Reviewing: {nb_name}')
        print("="*60)

        nb_path = Path(nb_name)
        if not nb_path.exists():
            print(f'⚠️  File not found: {nb_name}')
            continue

        code_cells = extract_code_cells(nb_path)
        print(f'Total code cells: {len(code_cells)}')

        issues = check_for_issues(code_cells)

        if issues['empty_functions']:
            print(f'\n⚠️  Empty functions found: {len(issues["empty_functions"])}')
            for cell_idx, snippet in issues['empty_functions'][:3]:
                print(f'  Cell {cell_idx}: {snippet}...')
        else:
            print('✓ No empty functions')

        if issues['unimplemented_todos']:
            print(f'\n⚠️  Unimplemented TODOs: {len(issues["unimplemented_todos"])}')
            for cell_idx, snippet in issues['unimplemented_todos'][:3]:
                print(f'  Cell {cell_idx}: {snippet}...')
        else:
            print('✓ No unimplemented TODOs')

        if issues['error_prone']:
            print(f'\n⚠️  Potential issues: {len(issues["error_prone"])}')
            for cell_idx, issue in issues['error_prone'][:3]:
                print(f'  Cell {cell_idx}: {issue}')
        else:
            print('✓ No obvious issues')

if __name__ == '__main__':
    main()
