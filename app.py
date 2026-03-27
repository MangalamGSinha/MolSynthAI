"""
MolSynthAI — AI-Powered Molecular Structure Generator
A Streamlit application that generates similar chemical structures using LLMs.
"""

import streamlit as st
import py3Dmol
from stmol import showmol
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, SDWriter, Descriptors, rdMolDescriptors
from rdkit.Chem.QED import qed
from io import BytesIO, StringIO
import json
import re
import tempfile
import os
import requests

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MolSynthAI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

/* ── CSS Variables: Dark Mode (default) ── */
:root {
    --bg-primary: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
    --bg-sidebar: linear-gradient(180deg, #13112b 0%, #1a1840 100%);
    --bg-card: rgba(255,255,255,0.04);
    --bg-header: linear-gradient(135deg, rgba(102, 51, 255, 0.15), rgba(0, 204, 255, 0.10));
    --border-card: rgba(255,255,255,0.08);
    --border-header: rgba(102, 51, 255, 0.25);
    --border-sidebar: rgba(102, 51, 255, 0.2);
    --text-primary: #e0e0ff;
    --text-secondary: rgba(255,255,255,0.65);
    --text-muted: rgba(255,255,255,0.45);
    --text-label: rgba(255,255,255,0.5);
    --sidebar-heading: #c4b5fd;
    --section-border: rgba(102, 51, 255, 0.3);
    --smiles-bg: rgba(0,204,255,0.08);
    --smiles-border: rgba(0,204,255,0.15);
    --smiles-color: #00ccff;
    --metric-bg: rgba(255,255,255,0.04);
    --metric-border: rgba(255,255,255,0.08);
    --viewer-bg: #1a1a3e;
}

/* ── CSS Variables: Light Mode ── */
@media (prefers-color-scheme: light) {
    :root {
        --bg-primary: #ffffff;
        --bg-sidebar: linear-gradient(180deg, #f8f7ff 0%, #eeedf8 100%);
        --bg-card: rgba(0,0,0,0.03);
        --bg-header: linear-gradient(135deg, rgba(102, 51, 255, 0.08), rgba(0, 204, 255, 0.06));
        --border-card: rgba(0,0,0,0.08);
        --border-header: rgba(102, 51, 255, 0.2);
        --border-sidebar: rgba(102, 51, 255, 0.15);
        --text-primary: #1a1a2e;
        --text-secondary: rgba(0,0,0,0.6);
        --text-muted: rgba(0,0,0,0.45);
        --text-label: rgba(0,0,0,0.5);
        --sidebar-heading: #5b21b6;
        --section-border: rgba(102, 51, 255, 0.2);
        --smiles-bg: rgba(0,100,180,0.06);
        --smiles-border: rgba(0,100,180,0.15);
        --smiles-color: #0066aa;
        --metric-bg: rgba(0,0,0,0.03);
        --metric-border: rgba(0,0,0,0.08);
        --viewer-bg: #ffffff;
    }
}

/* Molecule Image Adjustments for Dark Mode */
@media (prefers-color-scheme: dark) {
    .mol-card img, .input-preview img, [data-testid="stImage"] img {
        filter: invert(1) hue-rotate(180deg) brightness(1.3);
    }
}

/* Global */
html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Hide sidebar completely */
section[data-testid="stSidebar"] {
    display: none !important;
}
[data-testid="collapsedControl"] {
    display: none !important;
}
[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}

/* Main background */
.stApp {
    background: var(--bg-primary);
}

/* Header */
.main-header {
    background: var(--bg-header);
    border: 1px solid var(--border-header);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(12px);
    text-align: center;
}
.main-header h1 {
    background: linear-gradient(90deg, #6633ff, #00ccff, #00ffcc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: var(--text-secondary);
    font-size: 1.05rem;
    margin: 0.5rem 0 0 0;
    font-weight: 300;
}

/* Cards */
.mol-card {
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: 14px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    backdrop-filter: blur(8px);
}
.mol-card:hover {
    border-color: rgba(102, 51, 255, 0.4);
    box-shadow: 0 8px 32px rgba(102, 51, 255, 0.15);
    transform: translateY(-2px);
}
.mol-card .smiles-label {
    color: var(--smiles-color);
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    background: var(--smiles-bg);
    padding: 6px 12px;
    border-radius: 8px;
    word-break: break-all;
    margin-top: 0.6rem;
    border: 1px solid var(--smiles-border);
}

/* Section headers */
.section-header {
    color: var(--text-primary);
    font-size: 1.3rem;
    font-weight: 600;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--section-border);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Input molecule preview */
.input-preview {
    background: rgba(102, 51, 255, 0.08);
    border: 1px solid rgba(102, 51, 255, 0.2);
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    margin: 1rem 0;
}



/* Status badges */
.status-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-success {
    background: rgba(0, 200, 100, 0.12);
    color: #00aa55;
    border: 1px solid rgba(0, 200, 100, 0.25);
}
.badge-error {
    background: rgba(255, 68, 68, 0.12);
    color: #dd3333;
    border: 1px solid rgba(255, 68, 68, 0.25);
}
.badge-info {
    background: rgba(0, 150, 255, 0.12);
    color: var(--smiles-color);
    border: 1px solid rgba(0, 150, 255, 0.25);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6633ff, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 51, 255, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(102, 51, 255, 0.45) !important;
}

/* Download buttons */
.stDownloadButton > button {
    background: linear-gradient(135deg, #00ccff, #0099dd) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(0, 204, 255, 0.25) !important;
}
.stDownloadButton > button:hover {
    box-shadow: 0 6px 20px rgba(0, 204, 255, 0.4) !important;
}

/* Spinner */
.stSpinner > div {
    border-color: #6633ff transparent transparent transparent !important;
}

/* Metrics */
.metric-container {
    background: var(--metric-bg);
    border: 1px solid var(--metric-border);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #6633ff, #00ccff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    color: var(--text-label);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Property table */
.prop-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0 3px;
    font-size: 0.78rem;
    margin-top: 0.5rem;
}
.prop-table td {
    padding: 4px 8px;
}
.prop-table td:first-child {
    color: var(--text-muted);
    font-weight: 500;
    white-space: nowrap;
}
.prop-table td:last-child {
    color: var(--text-primary);
    font-weight: 600;
    text-align: right;
}
.prop-pass {
    color: #00cc66 !important;
}
.prop-fail {
    color: #ff5555 !important;
}
.prop-sim {
    background: linear-gradient(90deg, #6633ff, #00ccff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────
def sdf_to_smiles(sdf_content: bytes) -> str | None:
    """Convert SDF file content to SMILES using RDKit."""
    # Ensure content is bytes
    if isinstance(sdf_content, str):
        sdf_content = sdf_content.encode("utf-8")

    # Method 1: In-memory parsing with ForwardSDMolSupplier
    try:
        from rdkit.Chem import ForwardSDMolSupplier
        suppl = ForwardSDMolSupplier(BytesIO(sdf_content), removeHs=True, sanitize=True)
        for mol in suppl:
            if mol is not None:
                return Chem.MolToSmiles(mol)
    except Exception:
        pass

    # Method 2: Try without sanitization (some PubChem SDFs need this)
    try:
        from rdkit.Chem import ForwardSDMolSupplier
        suppl = ForwardSDMolSupplier(BytesIO(sdf_content), removeHs=True, sanitize=False)
        for mol in suppl:
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    pass
                return Chem.MolToSmiles(mol)
    except Exception:
        pass

    # Method 3: Temp file fallback
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".sdf", delete=False, mode="wb")
        tmp.write(sdf_content)
        tmp.close()
        suppl = Chem.SDMolSupplier(tmp.name, removeHs=True)
        for mol in suppl:
            if mol is not None:
                os.unlink(tmp.name)
                return Chem.MolToSmiles(mol)
        os.unlink(tmp.name)
    except Exception:
        pass

    # Method 4: Try parsing as a single mol block
    try:
        text = sdf_content.decode("utf-8", errors="ignore")
        mol_block = text.split("$$$$")[0].strip()
        if mol_block:
            mol = Chem.MolFromMolBlock(mol_block, removeHs=True)
            if mol is not None:
                return Chem.MolToSmiles(mol)
    except Exception:
        pass

    return None


def validate_smiles(smiles: str) -> Chem.Mol | None:
    """Return an RDKit Mol object if SMILES is valid, else None."""
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return mol
    except Exception:
        return None


def mol_to_3d(mol: Chem.Mol) -> str | None:
    """Generate 3D coordinates and return a Mol block string."""
    try:
        mol = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if result == -1:
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(), useRandomCoords=True)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        return Chem.MolToMolBlock(mol)
    except Exception:
        return None


def mol_to_svg(mol: Chem.Mol, size=(350, 280)) -> str:
    """Render a 2D depiction of a molecule as SVG with transparent background."""
    from rdkit.Chem import Draw
    drawer = Draw.MolDraw2DSVG(size[0], size[1])
    drawer.drawOptions().clearBackground = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def show_3d_viewer(mol_block: str, width: int = 340, height: int = 300):
    """Render a py3Dmol viewer with browser theme-aware background."""
    import streamlit.components.v1 as components
    # Escape backticks and backslashes in mol_block for JS template literal
    escaped_block = mol_block.replace("\\", "\\\\").replace("`", "\\`")
    html_content = f"""
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <div id="mol-viewer" style="width:{width}px;height:{height}px;position:relative;"></div>
    <script>
    (function() {{
        var isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        var bgColor = isDark ? '#1a1a3e' : '#ffffff';
        var viewer = $3Dmol.createViewer('mol-viewer', {{backgroundColor: bgColor}});
        viewer.addModel(`{escaped_block}`, 'mol');
        viewer.setStyle({{"stick": {{"radius": 0.12}}, "sphere": {{"scale": 0.25}}}});
        viewer.zoomTo();
        viewer.render();
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {{
            viewer.setBackgroundColor(e.matches ? '#1a1a3e' : '#ffffff');
            viewer.render();
        }});
    }})();
    </script>
    """
    components.html(html_content, height=height + 10, width=width + 10)


def compute_properties(mol: Chem.Mol, input_mol: Chem.Mol = None) -> dict:
    """Compute key molecular properties for a molecule."""
    props = {
        "MW": round(Descriptors.ExactMolWt(mol), 1),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "TPSA": round(Descriptors.TPSA(mol), 1),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "RotBonds": Descriptors.NumRotatableBonds(mol),
        "Rings": Descriptors.RingCount(mol),
        "AromaticRings": Descriptors.NumAromaticRings(mol),
        "FractionCSP3": round(Descriptors.FractionCSP3(mol), 2),
        "HeavyAtoms": mol.GetNumHeavyAtoms(),
        "QED": round(qed(mol), 3),
        "Formula": rdMolDescriptors.CalcMolFormula(mol),
    }
    # Lipinski compliance
    lipinski_violations = sum([
        props["MW"] > 500,
        props["LogP"] > 5,
        props["HBD"] > 5,
        props["HBA"] > 10,
    ])
    props["Lipinski"] = lipinski_violations

    # Tanimoto similarity to input molecule
    if input_mol is not None:
        try:
            fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
            fp1 = fpgen.GetFingerprint(input_mol)
            fp2 = fpgen.GetFingerprint(mol)
            props["Tanimoto"] = round(DataStructs.TanimotoSimilarity(fp1, fp2), 3)
        except Exception:
            props["Tanimoto"] = None
    else:
        props["Tanimoto"] = None

    return props


def props_to_html(props: dict) -> str:
    """Convert properties dict to a self-contained HTML string for components.html."""
    def _cls(val, limit, direction="le"):
        if direction == "le":
            return "prop-pass" if val <= limit else "prop-fail"
        return "prop-pass" if val >= limit else "prop-fail"

    lipinski_text = "✅ Pass" if props["Lipinski"] == 0 else f"⚠️ {props['Lipinski']} violation{'s' if props['Lipinski'] > 1 else ''}"
    lipinski_cls = "prop-pass" if props["Lipinski"] == 0 else "prop-fail"

    sim_html = ""
    if props.get("Tanimoto") is not None:
        sim_html = f'<tr><td class="prop-label">🔗 Similarity</td><td class="prop-val prop-sim">{props["Tanimoto"]:.3f}</td></tr>'

    return f"""
    <html><head><style>
    body {{ margin: 0; padding: 0; font-family: 'Inter', -apple-system, sans-serif; background: transparent; }}
    .prop-table {{ width: 100%; border-collapse: separate; border-spacing: 0 4px; font-size: 13px; }}
    .prop-label {{ color: rgba(255,255,255,1); font-weight: 500; white-space: nowrap; padding: 5px 8px; }}
    .prop-val {{ color: #e0e0ff; font-weight: 600; text-align: right; padding: 5px 8px; }}
    .prop-pass {{ color: #00cc66 !important; }}
    .prop-fail {{ color: #ff5555 !important; }}
    .prop-sim {{ background: linear-gradient(90deg, #6633ff, #00ccff); -webkit-background-clip: text;
                 -webkit-text-fill-color: transparent; font-weight: 700 !important; font-size: 14px !important; }}
    @media (prefers-color-scheme: light) {{
        .prop-label {{ color: rgba(0,0,0,0.85); }}
        .prop-val {{ color: #1a1a2e; }}
    }}
    </style></head><body>
    <table class="prop-table">
        {sim_html}
        <tr><td class="prop-label">📐 Formula</td><td class="prop-val">{props['Formula']}</td></tr>
        <tr><td class="prop-label">⚖️ MW</td><td class="prop-val {_cls(props['MW'], 500)}">{props['MW']}</td></tr>
        <tr><td class="prop-label">💧 LogP</td><td class="prop-val {_cls(props['LogP'], 5)}">{props['LogP']}</td></tr>
        <tr><td class="prop-label">🧲 TPSA</td><td class="prop-val">{props['TPSA']} Å²</td></tr>
        <tr><td class="prop-label">🤝 HBD / HBA</td><td class="prop-val"><span class="{_cls(props['HBD'], 5)}">{props['HBD']}</span> / <span class="{_cls(props['HBA'], 10)}">{props['HBA']}</span></td></tr>
        <tr><td class="prop-label">🔄 Rot. Bonds</td><td class="prop-val">{props['RotBonds']}</td></tr>
        <tr><td class="prop-label">💎 QED</td><td class="prop-val {_cls(props['QED'], 0.5, 'ge')}">{props['QED']}</td></tr>
        <tr><td class="prop-label">💊 Lipinski</td><td class="prop-val {lipinski_cls}">{lipinski_text}</td></tr>
    </table>
    </body></html>
    """


def render_props(props: dict, height: int = 310):
    """Render properties HTML using components.html for full HTML support."""
    import streamlit.components.v1 as components
    components.html(props_to_html(props), height=height, scrolling=False)


def build_prompt(smiles: str, num_molecules: int) -> str:
    """Build the LLM prompt to generate similar molecules."""
    return f"""You are an expert medicinal chemistry AI assistant specializing in lead optimization.
Your task is to generate structural analogs for the following input molecule in SMILES format: {smiles}

Generate exactly {num_molecules} novel, structurally distinct molecules designed for molecular docking and drug discovery. The generated molecules MUST abide by these rules:

1. Structural Similarity: Maintain the core scaffold or key pharmacophores of the input. Apply logical medicinal chemistry transformations (e.g., bioisosteric replacements, scaffold hopping, R-group sweeping, homologation, or ring expansions/contractions).
2. Chemical Validity & Feasibility: SMILES must be synthetically accessible and chemically valid. STRICTLY adhere to standard atomic valencies (e.g., Carbon=4; Nitrogen=3 or 4 with positive charge; Oxygen=2; Sulfur=2, 4, or 6; Phosphorus=3 or 5; Hydrogen=1; Halogens (F, Cl, Br, I)=1). Avoid Texas carbons and impossible oxidation states.
3. Drug-likeness: Aim for Lipinski's Rule of Five compliance (MW ≤ 500, LogP ≤ 5, H-bond donors ≤ 5, H-bond acceptors ≤ 10). Avoid known PAINS (Pan Assay Interference Compounds) and highly reactive functional groups.
4. Diversity: Each generated molecule must be structurally unique from the others in the set.

Return ONLY a valid JSON array of valid SMILES strings. Do not include any explanations, markdown code blocks, or numbering.
Example output format: ["CCO", "CCCO", "CC(O)C"]

Your response must be ONLY the JSON array. Output nothing else."""


def call_openai(api_key: str, prompt: str) -> str:
    """Call OpenAI API and return the response text."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a medicinal chemistry expert. Always respond with only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
        max_tokens=2000,
    )
    return response.choices[0].message.content.strip()


def call_gemini(api_keys: list[str], prompt: str) -> str:
    """Call Gemini API with automatic key failover.
    
    Tries each API key in order. If a key fails due to quota exhaustion
    or rate limiting, moves to the next key. Raises an error only if
    all keys are exhausted.
    """
    import google.generativeai as genai

    # Start from the last known working key index
    start_idx = st.session_state.get("_gemini_key_idx", 0)
    n = len(api_keys)
    last_error = None

    for attempt in range(n):
        idx = (start_idx + attempt) % n
        key = api_keys[idx]
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            # Success — remember this key index for next time
            st.session_state["_gemini_key_idx"] = idx
            return response.text.strip()
        except Exception as e:
            err_str = str(e).lower()
            is_quota_error = any(kw in err_str for kw in [
                "429", "quota", "rate", "resource_exhausted",
                "resourceexhausted", "limit", "exceeded",
            ])
            last_error = e
            if is_quota_error and attempt < n - 1:
                # This key is exhausted, try the next one
                st.toast(f"⚠️ API key #{idx + 1} quota hit — switching to key #{(idx + 1) % n + 1}...", icon="🔄")
                continue
            else:
                raise

    raise last_error  # All keys exhausted


def parse_smiles_response(response_text: str) -> list[str]:
    """Parse the LLM response to extract SMILES strings."""
    # Clean up markdown code blocks if the LLM wrapped it
    response_text = response_text.replace("```json", "").replace("```", "").strip()

    # Try direct JSON parse
    try:
        result = json.loads(response_text)
        if isinstance(result, list):
            return [s.strip() for s in result if isinstance(s, str)]
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in the response
    match = re.search(r'\[.*?\]', response_text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return [s.strip() for s in result if isinstance(s, str)]
        except json.JSONDecodeError:
            pass

    # Fallback: extract lines that look like SMILES
    lines = response_text.split('\n')
    smiles_list = []
    for line in lines:
        line = line.strip().strip('"').strip("'").strip(',').strip('"').strip("'")
        if line and not line.startswith('{') and not line.startswith('['):
            mol = validate_smiles(line)
            if mol is not None:
                smiles_list.append(line)
    return smiles_list


def generate_sdf_download(mols: list[tuple[str, Chem.Mol]]) -> bytes:
    """Create a multi-molecule SDF file from a list of (smiles, mol) tuples."""
    output = StringIO()
    writer = SDWriter(output)
    for smiles, mol in mols:
        mol_3d = Chem.AddHs(mol)
        try:
            AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=200)
        except Exception:
            pass
        mol_3d.SetProp("SMILES", smiles)
        writer.write(mol_3d)
    writer.close()
    return output.getvalue().encode("utf-8")


def generate_smi_download(smiles_list: list[str]) -> str:
    """Create a .smi file content from a list of SMILES strings."""
    lines = []
    for i, smi in enumerate(smiles_list, 1):
        lines.append(f"{smi}\tMolSynth_{i}")
    return "\n".join(lines)


def generate_zip_sdf(mols: list[tuple[str, Chem.Mol]]) -> bytes:
    """Create a ZIP file containing individual SDF files."""
    import zipfile
    import io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, (smi, mol) in enumerate(mols, 1):
            sdf_data = generate_sdf_download([(smi, mol)])
            zf.writestr(f"MolSynth_{i}.sdf", sdf_data)
    return buf.getvalue()


def generate_zip_smi(smiles_list: list[str]) -> bytes:
    """Create a ZIP file containing individual SMI files."""
    import zipfile
    import io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, smi in enumerate(smiles_list, 1):
            smi_data = f"{smi}\tMolSynth_{i}"
            zf.writestr(f"MolSynth_{i}.smi", smi_data)
    return buf.getvalue()


def generate_csv_download(mols: list[tuple[str, 'Chem.Mol']], input_mol: 'Chem.Mol' = None) -> str:
    """Create a CSV string with SMILES and all computed properties."""
    import csv
    import io
    buf = io.StringIO()
    fieldnames = ["ID", "SMILES", "Formula", "MW", "LogP", "TPSA",
                  "HBD", "HBA", "RotBonds", "Rings", "AromaticRings",
                  "FractionCSP3", "HeavyAtoms", "QED", "Lipinski_Violations", "Tanimoto_Similarity"]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for i, (smi, mol) in enumerate(mols, 1):
        props = compute_properties(mol, input_mol)
        writer.writerow({
            "ID": f"MolSynth_{i}",
            "SMILES": smi,
            "Formula": props["Formula"],
            "MW": props["MW"],
            "LogP": props["LogP"],
            "TPSA": props["TPSA"],
            "HBD": props["HBD"],
            "HBA": props["HBA"],
            "RotBonds": props["RotBonds"],
            "Rings": props["Rings"],
            "AromaticRings": props["AromaticRings"],
            "FractionCSP3": props["FractionCSP3"],
            "HeavyAtoms": props["HeavyAtoms"],
            "QED": props["QED"],
            "Lipinski_Violations": props["Lipinski"],
            "Tanimoto_Similarity": props.get("Tanimoto", ""),
        })
    return buf.getvalue()


# ─────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧬 MolSynthAI</h1>
    <p>Generate structurally similar molecules for docking &amp; lead optimization — powered by AI</p>
</div>
""", unsafe_allow_html=True)

# ── Molecule Input ──
st.markdown('<div class="section-header">📥 Input Molecule</div>', unsafe_allow_html=True)

input_smiles = None
input_mol = None

# Sample molecules
SAMPLE_MOLECULES = {
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    # "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    # "Penicillin G": "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",
}

# Callback for sample buttons
def set_sample(smi):
    st.session_state["_pending_smiles"] = smi

# Apply pending sample to the field before widget renders
if "_pending_smiles" in st.session_state:
    st.session_state["smiles_field"] = st.session_state.pop("_pending_smiles")

# Sample SMILES buttons
st.markdown('<p style="color: var(--text-muted); font-size:0.82rem; margin-bottom:0.4rem;">🧪 Try a sample:</p>', unsafe_allow_html=True)
sample_cols = st.columns(len(SAMPLE_MOLECULES))
for idx, (name, smi) in enumerate(SAMPLE_MOLECULES.items()):
    with sample_cols[idx]:
        st.button(name, key=f"sample_{name}", width="stretch", on_click=set_sample, args=(smi,))

# Input field + submit button
input_col, btn_col = st.columns([5, 1])
with input_col:
    smiles_input = st.text_input(
        "Enter SMILES string",
        placeholder="Enter SMILES string",
        label_visibility="collapsed",
        key="smiles_field",
    )
with btn_col:
    submit_clicked = st.button("🔍 Submit", width="stretch")

# Validate SMILES only on submit
if submit_clicked and smiles_input:
    mol = validate_smiles(smiles_input)
    if mol:
        st.session_state["submitted_smiles"] = Chem.MolToSmiles(mol)
        st.session_state["submitted_mol"] = mol
    else:
        st.session_state.pop("submitted_smiles", None)
        st.session_state.pop("submitted_mol", None)
        st.markdown('<span class="status-badge badge-error">✗ Invalid SMILES</span>', unsafe_allow_html=True)

if "submitted_smiles" in st.session_state and "submitted_mol" in st.session_state:
    input_smiles = st.session_state["submitted_smiles"]
    input_mol = st.session_state["submitted_mol"]

# ── Show input molecule preview ──
if input_mol is not None:
    st.markdown('<div class="section-header">🔍 Input Molecule Preview</div>', unsafe_allow_html=True)

    col_2d, col_3d, col_info = st.columns([2, 3, 2])

    with col_2d:
        st.markdown("**2D Structure**")
        svg_text = mol_to_svg(input_mol, size=(400, 320))
        st.image(svg_text, width="stretch")

    with col_3d:
        st.markdown("**3D Structure**")
        mol_block = mol_to_3d(input_mol)
        if mol_block:
            show_3d_viewer(mol_block, width=500, height=350)
        else:
            st.warning("Could not generate 3D coordinates.")

    with col_info:
        st.markdown("**Molecule Info**")
        input_props = compute_properties(input_mol)
        render_props(input_props, height=300)


# ── Generate Button ──
st.markdown("---")

col_slider1, col_slider2, col_slider3 = st.columns([1, 2, 1])
with col_slider2:
    num_molecules = st.slider(
        "🎛️ Number of molecules to generate",
        min_value=1,
        max_value=30,
        value=10,
        step=1,
    )

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    generate_clicked = st.button(
        f"🚀 Generate {num_molecules} Similar Molecules",
        width="stretch",
        disabled=(input_mol is None),
    )

if input_mol is None:
    st.markdown(
        '<p style="text-align:center; color:rgba(255,255,255,0.4); font-size:0.9rem;">'
        '↑ Enter a molecule above to get started</p>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Generation Logic
# ─────────────────────────────────────────────
api_keys = list(st.secrets["GEMINI_API_KEYS"])

if generate_clicked and input_mol is not None:
    prompt = build_prompt(input_smiles, num_molecules)

    with st.spinner("🧪 Generating molecules with AI..."):
        try:
            raw_response = call_gemini(api_keys, prompt)

            smiles_list = parse_smiles_response(raw_response)

            # Validate each SMILES
            valid_mols = []
            for smi in smiles_list:
                mol = validate_smiles(smi)
                if mol is not None:
                    canonical = Chem.MolToSmiles(mol)
                    # Skip duplicates and the input molecule itself
                    if canonical != input_smiles and canonical not in [s for s, _ in valid_mols]:
                        valid_mols.append((canonical, mol))

            st.session_state["generated_mols"] = valid_mols
            st.session_state["generation_done"] = True

        except Exception as e:
            st.error(f"❌ Error during generation: {str(e)}")
            st.session_state["generation_done"] = False


# ─────────────────────────────────────────────
# Results Display
# ─────────────────────────────────────────────
if st.session_state.get("generation_done") and st.session_state.get("generated_mols"):
    valid_mols = st.session_state["generated_mols"]

    st.markdown(f"""
    <div class="section-header">
        🧪 Generated Molecules
        <span class="status-badge badge-success" style="margin-left: auto; font-size: 0.75rem;">
            {len(valid_mols)} valid molecules
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Summary metrics ──
    mcols = st.columns(3)
    with mcols[0]:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{len(valid_mols)}</div>
            <div class="metric-label">Valid Molecules</div>
        </div>
        """, unsafe_allow_html=True)
    with mcols[1]:
        from rdkit.Chem import Descriptors as Desc
        avg_mw = sum(Desc.ExactMolWt(m) for _, m in valid_mols) / len(valid_mols) if valid_mols else 0
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{avg_mw:.1f}</div>
            <div class="metric-label">Avg Mol Weight</div>
        </div>
        """, unsafe_allow_html=True)
    with mcols[2]:
        avg_logp = sum(Desc.MolLogP(m) for _, m in valid_mols) / len(valid_mols) if valid_mols else 0
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{avg_logp:.2f}</div>
            <div class="metric-label">Avg LogP</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Download buttons ──
    st.markdown("**Export All Results**")
    
    # 1. Summary Data
    csv_data = generate_csv_download(valid_mols, input_mol)
    st.download_button(
        label="⬇️ Download All Properties & SMILES (CSV)",
        data=csv_data,
        file_name="molsynthai_generated.csv",
        mime="text/csv",
        width="stretch",
    )
    
    st.markdown("<div style='margin-top: 1rem; margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("<span style='font-size:0.85rem; font-weight:600; color:var(--text-secondary);'>Structural Formats</span>", unsafe_allow_html=True)
    
    export_format = st.radio(
        "Export Format",
        ["Combined File", "ZIP of Individual Files"],
        horizontal=True,
        label_visibility="collapsed",
    )

    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        if export_format == "Combined File":
            dl_data_sdf = generate_sdf_download(valid_mols)
            file_name_sdf = "molsynthai_generated.sdf"
            mime_sdf = "chemical/x-mdl-sdfile"
        else:
            dl_data_sdf = generate_zip_sdf(valid_mols)
            file_name_sdf = "molsynthai_generated_sdf.zip"
            mime_sdf = "application/zip"

        st.download_button(
            label="⬇️ Download All (SDF)",
            data=dl_data_sdf,
            file_name=file_name_sdf,
            mime=mime_sdf,
            width="stretch",
        )

    with dl_col2:
        if export_format == "Combined File":
            dl_data_smi = generate_smi_download([s for s, _ in valid_mols])
            file_name_smi = "molsynthai_generated.smi"
            mime_smi = "text/plain"
        else:
            dl_data_smi = generate_zip_smi([s for s, _ in valid_mols])
            file_name_smi = "molsynthai_generated_smi.zip"
            mime_smi = "application/zip"

        st.download_button(
            label="⬇️ Download All (SMI)",
            data=dl_data_smi,
            file_name=file_name_smi,
            mime=mime_smi,
            width="stretch",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── View toggle ──
    view_mode = st.radio(
        "View mode",
        ["🖼️ 2D Structures", "🔬 3D Interactive"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # ── Molecule Grid ──
    cols_per_row = 3
    for i in range(0, len(valid_mols), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(valid_mols):
                break
            smi, mol = valid_mols[idx]

            with col:
              with st.container(border=True):
                st.markdown(
                    f'<span class="status-badge badge-info" style="margin-bottom:0.5rem; display:inline-block;">'
                    f'Mol #{idx + 1}</span>',
                    unsafe_allow_html=True,
                )

                if "2D" in view_mode:
                    svg_text = mol_to_svg(mol, size=(350, 280))
                    st.image(svg_text, width="stretch")
                else:
                    mol_block = mol_to_3d(mol)
                    if mol_block:
                        show_3d_viewer(mol_block, width=340, height=280)
                    else:
                        st.warning("3D generation failed")

                st.markdown(f'<div class="smiles-label">{smi}</div>', unsafe_allow_html=True)

                # Properties
                props = compute_properties(mol, input_mol)
                st.markdown('<p style="color: var(--text-secondary); font-size: 0.85rem; margin: 0.6rem 0 0 0; font-weight: 600;">📊 Properties</p>', unsafe_allow_html=True)
                render_props(props, height=330)

                st.markdown("<div style='margin-top: 0.4rem;'></div>", unsafe_allow_html=True)
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    st.download_button(
                        label="⬇️ SDF",
                        data=generate_sdf_download([(smi, mol)]),
                        file_name=f"MolSynth_{idx+1}.sdf",
                        mime="chemical/x-mdl-sdfile",
                        key=f"dl_sdf_{idx}",
                        width="stretch"
                    )
                with dl_col2:
                    st.download_button(
                        label="⬇️ SMI",
                        data=f"{smi}\tMolSynth_{idx+1}",
                        file_name=f"MolSynth_{idx+1}.smi",
                        mime="text/plain",
                        key=f"dl_smi_{idx}",
                        width="stretch"
                    )
                


elif st.session_state.get("generation_done") and not st.session_state.get("generated_mols"):
    st.warning("⚠️ No valid molecules were generated. Try again or use a different input molecule.")


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")

st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(255, 170, 0, 0.08), rgba(255, 100, 0, 0.06));
    border: 1px solid rgba(255, 170, 0, 0.25);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
">
    <span style="font-size: 1.5rem; line-height: 1;">⚠️</span>
    <div>
        <strong style="color: #ffaa00; font-size: 0.95rem;">AI-Generated Results — Use With Caution</strong>
        <p style="color: var(--text-secondary); font-size: 0.85rem; margin: 0.4rem 0 0 0; line-height: 1.6;">
            The molecules generated by MolSynthAI are predicted by artificial intelligence and have <strong>not been experimentally validated</strong>.
            Generated compounds may not be synthesizable, may not exist in nature, and could have unpredictable chemical properties.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: 14px;
    padding: 2rem 2.5rem;
    text-align: center;
    margin-bottom: 1rem;
">
    <p style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 1.2rem;">
        Built With
    </p>
    <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 0.8rem; margin-bottom: 1.5rem;">
        <span style="background: rgba(102, 51, 255, 0.12); color: #a78bfa; border: 1px solid rgba(102, 51, 255, 0.25); padding: 6px 16px; border-radius: 20px; font-size: 0.82rem; font-weight: 500;">
            🐍 Python
        </span>
        <span style="background: rgba(255, 75, 75, 0.10); color: #ff6b6b; border: 1px solid rgba(255, 75, 75, 0.25); padding: 6px 16px; border-radius: 20px; font-size: 0.82rem; font-weight: 500;">
            🎈 Streamlit
        </span>
        <span style="background: rgba(0, 200, 100, 0.10); color: #00cc66; border: 1px solid rgba(0, 200, 100, 0.25); padding: 6px 16px; border-radius: 20px; font-size: 0.82rem; font-weight: 500;">
            🧪 RDKit
        </span>
        <span style="background: rgba(0, 150, 255, 0.10); color: #4da6ff; border: 1px solid rgba(0, 150, 255, 0.25); padding: 6px 16px; border-radius: 20px; font-size: 0.82rem; font-weight: 500;">
            🤖 Google Gemini
        </span>
        <span style="background: rgba(255, 200, 0, 0.10); color: #ffcc33; border: 1px solid rgba(255, 200, 0, 0.25); padding: 6px 16px; border-radius: 20px; font-size: 0.82rem; font-weight: 500;">
            🔬 py3Dmol
        </span>
    </div>
    <p style="color: var(--text-muted); font-size: 0.78rem; margin: 0; line-height: 1.6;">
        <strong style="color: var(--text-secondary);">MolSynthAI</strong> leverages large language models for molecular generation,
        RDKit for cheminformatics & 2D depiction, py3Dmol for interactive 3D visualization,
        and Streamlit for the web interface.
    </p>
    <p style="color: var(--text-muted); font-size: 0.7rem; margin-top: 1rem;">
        © 2026 MolSynthAI · For Research & Educational Use Only
    </p>
</div>
""", unsafe_allow_html=True)
