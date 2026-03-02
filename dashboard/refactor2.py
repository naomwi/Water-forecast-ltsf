import sys

def modify_app():
    with open('c:\\Users\\admin\\Documents\\Project\\weather_ltsf\\dashboard\\app.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Add session state initialization
    init_block = '''
# ==========================================
# SESSION STATE NAVIGATION
# ==========================================
if "current_page" not in st.session_state:
    st.session_state.current_page = "Chat"

def set_page(page_name):
    st.session_state.current_page = page_name
'''

    content = content.replace('# Init model once\ninit_gemini()', init_block + '\n# Init model once\ninit_gemini()')

    # 2. Extract the expander blocks
    start_idx_us = content.find('    with st.expander("ℹ️ About Us"):')
    end_idx_us = content.find('    with st.expander("📖 About the Project"):')
    end_idx_proj = content.find('        </div>\n        """, unsafe_allow_html=True)\n\n\n# ==========================================')

    if start_idx_us != -1 and end_idx_us != -1 and end_idx_proj != -1:
        end_idx_proj += len('        </div>\n        """, unsafe_allow_html=True)\n')
        
        us_block = content[start_idx_us:end_idx_us]
        proj_block = content[end_idx_us:end_idx_proj]
        
        # Extract inner components
        team_start = us_block.find('        # ---- Team Members ----')
        team_end = us_block.find('        <hr class="section-divider">')
        team_comp = us_block[team_start:team_end].replace('        ', '    ') # adjust indent
        
        methodology_start = proj_block.find('        # ---- Methodology ----')
        models_start = proj_block.find('        # ---- Model Comparison ----')
        method_comp = proj_block[methodology_start:models_start].replace('        ', '    ')
        models_comp = proj_block[models_start:].replace('        ', '    ')
        
        # Remove existing expanders from content
        content = content[:start_idx_us] + content[end_idx_proj:]
        
        # 3. Add to Sidebar the navigation button and fixed Github footer
        sidebar_nav_css = '''
    <style>
    .github-footer {
        position: fixed;
        bottom: 24px;
        left: 20px;
        z-index: 1000;
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 14px;
        background: rgba(255,123,0,0.1);
        border: 1px solid rgba(255,123,0,0.2);
        border-radius: 8px;
        text-decoration: none;
        transition: all 0.2s ease;
    }
    .github-footer:hover {
        background: rgba(255,123,0,0.2);
        border-color: #ff7b00;
        transform: translateY(-2px);
    }
    .github-footer svg {
        fill: #ff7b00;
        width: 18px;
        height: 18px;
    }
    .github-footer span {
        color: #ff7b00;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    <a href="https://github.com/TrumAIFPTU/hydropred" target="_blank" class="github-footer">
        <svg viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>
        <span>TrumAIFPTU/hydropred</span>
    </a>
    '''
        
        # Find insert for sidebar buttons
        target_sb_line = '        <span>Methodology, experimental results, and analysis from the team\\'s capstone report.</span>\n    </div>\n    """, unsafe_allow_html=True)\n'
        sb_insert = content.find(target_sb_line)
        if sb_insert != -1:
            sb_button_code = '''
    st.markdown("<br>", unsafe_allow_html=True)
    if st.session_state.current_page == "Chat":
        st.button("📖 About the Project", use_container_width=True, on_click=set_page, args=("About",))
    else:
        st.button("💬 Back to Chat", use_container_width=True, on_click=set_page, args=("Chat",))
        
    st.markdown("""
''' + sidebar_nav_css + '\n    """, unsafe_allow_html=True)\n'
            
            content = content.replace(target_sb_line, target_sb_line + sb_button_code)
        
        # 4. Handle Main View Routing
        main_section = content.find('# MAIN CONTENT')
        chat_init = content.find('if "chat_history" not in st.session_state')
        
        if main_section != -1 and chat_init != -1:
            # Fix the CSS grid widths for full page
            method_comp = method_comp.replace('grid-template-columns: 1fr;', 'grid-template-columns: repeat(3, 1fr);')
            models_comp = models_comp.replace('grid-template-columns: 1fr;', 'grid-template-columns: repeat(3, 1fr);')
            team_comp = team_comp.replace('grid-template-columns: 1fr;', 'grid-template-columns: repeat(5, 1fr);')
            
            about_view_html = '''
    st.markdown("""
    <style>
    .about-header {
        text-align: center;
        padding: 40px 0 20px;
        animation: fadeUp 0.6s ease-out;
    }
    .about-header h1 {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff, #a3a3a3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 12px;
    }
    .about-header p {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.6);
        max-width: 600px;
        margin: 0 auto;
    }
    </style>
    
    <div class="about-header">
        <h1>About the Project</h1>
        <p>A deep dive into the methodology, architectures, and the team behind HydroPred AI.</p>
    </div>
    """, unsafe_allow_html=True)
'''
            
            about_view = f'''# ==========================================
# ABOUT PAGE VIEW
# ==========================================
if st.session_state.current_page == "About":
{about_view_html}
{method_comp}
{models_comp}
{team_comp}
'''
            chat_content = content[chat_init:]
            indented_chat = []
            for line in chat_content.split('\n'):
                if line.strip():
                    indented_chat.append('    ' + line)
                else:
                    indented_chat.append(line)
            chat_block = 'elif st.session_state.current_page == "Chat":\n' + '\n'.join(indented_chat)
            
            content = content[:chat_init] + about_view + '\n' + chat_block

        with open('c:\\Users\\admin\\Documents\\Project\\weather_ltsf\\dashboard\\app.py', 'w', encoding='utf-8') as f:
            f.write(content)

        print('Successfully applied full-page refactoring!')
    else:
        print('Failed to find expander blocks.')

modify_app()
