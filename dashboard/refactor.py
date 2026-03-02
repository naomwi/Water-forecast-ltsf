import sys

def modify_app():
    with open('c:\\Users\\admin\\Documents\\Project\\weather_ltsf\\dashboard\\app.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Lines 468-568 are index 467 to 568
    to_move = lines[467:568]
    
    # We increase indentation by 4 spaces for the moved lines
    indented_move = []
    for line in to_move:
        if line.strip() == '':
            indented_move.append(line)
        else:
            indented_move.append('    ' + line)

    # Remove the moved block from the original list (backwards to avoid index shifting)
    del lines[467:568]
    
    # Find the right insertion point dynamically 
    insert_idx = -1
    for i, line in enumerate(lines):
        if '    \"\"\", unsafe_allow_html=True)' in line and '    <div class=\"sb-card\">' in ''.join(lines[i-15:i]):
            insert_idx = i
            break
            
    if insert_idx == -1:
        insert_idx = 269
        
    expander_code = [
        '\n',
        '    with st.expander(\"ℹ️ About Us\"):\n'
    ]
    
    lines = lines[:insert_idx+1] + expander_code + indented_move + lines[insert_idx+1:]
    
    content = ''.join(lines)
    content = content.replace('grid-template-columns: repeat(3, 1fr);\n        gap: 12px;\n        margin-bottom: 24px;', 'grid-template-columns: 1fr;\n        gap: 12px;\n        margin-bottom: 24px;')
    content = content.replace('grid-template-columns: repeat(3, 1fr);\n        gap: 12px;\n        margin-bottom: 20px;', 'grid-template-columns: 1fr;\n        gap: 12px;\n        margin-bottom: 20px;')
    content = content.replace('grid-template-columns: repeat(5, 1fr);\n        gap: 12px;\n        margin-bottom: 24px;', 'grid-template-columns: 1fr;\n        gap: 12px;\n        margin-bottom: 24px;')

    content = content.replace('grid-template-columns: repeat(3, 1fr);', 'grid-template-columns: 1fr;')
    content = content.replace('grid-template-columns: repeat(5, 1fr);', 'grid-template-columns: 1fr;')
    
    content = content.replace('<div class=\"section-heading\">🔬 Methodology Pipeline</div>', '<div class=\"section-heading\" style=\"margin-top:0;\">🔬 Methodology Pipeline</div>')

    with open('c:\\Users\\admin\\Documents\\Project\\weather_ltsf\\dashboard\\app.py', 'w', encoding='utf-8') as f:
        f.write(content)
        
    print('Successfully moved block to sidebar with Python.')

modify_app()
