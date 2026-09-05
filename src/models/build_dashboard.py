"""
Build the interactive HTML dashboard for Beyond the Bell Curve (Plotly).

Non-technical-friendly version:
  - Charts reworked to be readable by a general audience (meaning over jargon).
  - Real model names + scores on hover; a "what are these models" explainer box.
  - Country explorer shows a short description + recorded crisis years.
  - Hover tooltips on technical terms + a glossary box.
  - Runs through 2023 (model trained on <=2016, applied to fresh data).

INPUT (outputs/): feature_importance_permutation.csv, model_results.csv,
                  robustness_per_year.csv, country_risk_2023.csv
INPUT (project root): country_info.json
OUTPUT: outputs/dashboard.html
"""

import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html

OUT = Path("outputs")
ROOT = Path(".")

COL_TAIL = "#d1495b"
COL_CONV = "#3a6ea5"
COL_MACRO = "#8a8d91"
DASH = "\u2014"

PRETTY = {
    'dep_1m': "Recent 1-month drop", 'dep_mean_3m': "Average drop (3 mo)",
    'dep_mean_6m': "Average drop (6 mo)", 'dep_mean_12m': "Average drop (12 mo)",
    'dep_vol_3m': "Currency choppiness (3 mo)", 'dep_vol_6m': "Currency choppiness (6 mo)",
    'dep_vol_12m': "Currency choppiness (12 mo)", 'dep_cum_3m': "Total drop (3 mo)",
    'dep_cum_6m': "Total drop (6 mo)", 'dep_cum_12m': "Total drop (12 mo)",
    'dep_skew_6m': "Lopsided toward crashes (6 mo)",
    'dep_skew_12m': "Lopsided toward crashes (12 mo)",
    'dep_kurt_6m': "Fat-tail extremeness (6 mo)", 'dep_kurt_12m': "Fat-tail extremeness (12 mo)",
    'dep_max_6m': "Worst single month (6 mo)", 'dep_max_12m': "Worst single month (12 mo)",
    'dep_es90_6m': "Average of worst months (6 mo)", 'dep_es90_12m': "Average of worst months (12 mo)",
    'extreme_count_3m': "Count of violent drops (3 mo)",
    'extreme_count_6m': "Count of violent drops (6 mo)",
    'extreme_count_12m': "Count of violent drops (12 mo)",
}
GROUP_LABEL = {'tail': 'Extreme-move clue', 'conventional': 'Ordinary clue', 'macro': 'Economy clue'}
GROUP_COLOR = {'tail': COL_TAIL, 'conventional': COL_CONV, 'macro': COL_MACRO}


def fig_importance():
    imp = pd.read_csv(OUT / "feature_importance_permutation.csv")
    pos = imp[imp['importance'] > 0].copy()
    total = pos['importance'].sum()
    pos['share'] = 100 * pos['importance'] / total
    top = pos.sort_values('share', ascending=False).head(10)
    fig = go.Figure(go.Bar(
        x=top['share'][::-1], y=[PRETTY.get(f, f) for f in top['feature'][::-1]],
        orientation='h',
        marker_color=[GROUP_COLOR.get(g, '#ccc') for g in top['group'][::-1]],
        customdata=[GROUP_LABEL.get(g, g) for g in top['group'][::-1]],
        hovertemplate="%{y}<br>%{customdata}<br>share of the signal: %{x:.0f}%<extra></extra>",
        text=[f"{v:.0f}%" for v in top['share'][::-1]], textposition='outside'))
    shares = 100 * pos.groupby('group')['importance'].sum() / total
    sub = (f"<span style='color:{COL_TAIL}'>&#9679; Extreme-move clues {shares.get('tail',0):.0f}%</span>   "
           f"<span style='color:{COL_MACRO}'>&#9679; Economy clues {shares.get('macro',0):.0f}%</span>   "
           f"<span style='color:{COL_CONV}'>&#9679; Ordinary clues {shares.get('conventional',0):.0f}%</span>")
    fig.update_layout(title=f"Which warning signs matter most?<br><sub>{sub}</sub>",
                      xaxis_title="Share of the model's warning power (%)",
                      template='plotly_white', height=460, margin=dict(l=10, r=60, t=80, b=45))
    return fig


def fig_models():
    m = pd.read_csv(OUT / "model_results.csv")
    models = ['Logistic', 'RandomForest', 'XGBoost']
    nice = {'Logistic': 'Simple model', 'RandomForest': 'Medium model', 'XGBoost': 'Complex model'}
    realname = {'Logistic': 'Logistic Regression', 'RandomForest': 'Random Forest', 'XGBoost': 'XGBoost'}
    improve, hover = [], []
    for x in models:
        c = float(m[(m.model == x) & (m.features == 'Conventional')]['pr_auc'].iloc[0])
        t = float(m[(m.model == x) & (m.features == 'Conv+Tail')]['pr_auc'].iloc[0])
        improve.append(100 * (t - c) / c)
        hover.append(f"{realname[x]}<br>score with ordinary clues: {c:.3f}"
                     f"<br>with extreme-move clues added: {t:.3f}")
    fig = go.Figure(go.Bar(
        x=[nice[x] for x in models], y=improve,
        marker_color=[COL_TAIL if v > 0 else COL_MACRO for v in improve],
        text=[f"{v:+.0f}%" for v in improve], textposition='outside',
        customdata=hover, hovertemplate="%{customdata}<br><b>improvement: %{y:+.0f}%</b><extra></extra>"))
    fig.update_layout(title="Do the extreme-move clues actually help?<br>"
                      "<sub>Improvement in crisis-catching when the extreme-move clues are added</sub>",
                      yaxis_title="Improvement (%)", template='plotly_white', height=430,
                      margin=dict(l=10, r=10, t=80, b=40))
    return fig


def fig_peryear():
    r = pd.read_csv(OUT / "robustness_per_year.csv")
    helped = int((r['diff'] > 0).sum()); tot = len(r)
    colors = [COL_TAIL if d > 0 else COL_MACRO for d in r['diff']]
    fig = go.Figure(go.Bar(
        x=r['year'].astype(str), y=r['diff'], marker_color=colors,
        hovertemplate="%{x}: %{customdata}<extra></extra>",
        customdata=["helped" if d > 0 else "didn't help" for d in r['diff']]))
    fig.update_layout(title=f"Was it a fluke? Extreme-move clues helped in {helped} of {tot} years<br>"
                      "<sub>Red bar = they helped that year</sub>",
                      yaxis_title="Improvement that year", yaxis=dict(showticklabels=False),
                      template='plotly_white', height=430, margin=dict(l=10, r=10, t=80, b=40))
    return fig


def load_country_data():
    pred = pd.read_csv(OUT / "country_risk_2023.csv")
    pred['date'] = pd.to_datetime(pred['date'])
    latest = pred.sort_values('date').groupby('iso3').tail(1).copy()
    latest['rank'] = latest['risk_raw'].rank(ascending=False, method='min').astype(int)
    n = len(latest)
    latest['pct_rank'] = 100 * (1 - (latest['rank'] - 1) / (n - 1))
    latest['last_date'] = latest['date'].dt.strftime('%b %Y')
    rank_map = latest.set_index('iso3')[['rank', 'pct_rank', 'risk_calibrated', 'last_date']].to_dict('index')
    return pred, rank_map, n


def build_country_figs(pred):
    figs = {}
    for iso, g in pred.groupby('iso3'):
        g = g.sort_values('date')
        fig = go.Figure()
        fig.add_scatter(x=g['date'], y=100 * g['risk_calibrated'], mode='lines',
                        line=dict(color=COL_CONV, width=1.5), name='Estimated risk',
                        hovertemplate="%{x|%Y-%m}: %{y:.1f}%<extra></extra>")
        cr = g[g['actual_target'] == 1]
        if len(cr):
            fig.add_scatter(x=cr['date'], y=100 * cr['risk_calibrated'], mode='markers',
                            marker=dict(color=COL_TAIL, size=6),
                            name='Real crisis window (to 2016)',
                            hovertemplate="crisis window %{x|%Y-%m}<extra></extra>")
        fig.update_layout(template='plotly_white', height=300, yaxis_title="Estimated risk (%)",
                          margin=dict(l=10, r=10, t=10, b=30),
                          legend=dict(orientation='h', y=-0.25, x=0.5, xanchor='center'))
        figs[iso] = to_html(fig, include_plotlyjs=False, full_html=False, div_id=f"tl_{iso}")
    return figs


CNAMES = {
    'ARG':'Argentina','BRA':'Brazil','TUR':'Turkey','KOR':'South Korea','MEX':'Mexico',
    'RUS':'Russia','IDN':'Indonesia','ZAF':'South Africa','EGY':'Egypt','NGA':'Nigeria',
    'AGO':'Angola','COD':'DR Congo','BGD':'Bangladesh','DOM':'Dominican Rep.','UZB':'Uzbekistan',
    'BLR':'Belarus','GIN':'Guinea','SUR':'Suriname','NAM':'Namibia','SWZ':'Eswatini',
    'LSO':'Lesotho','VEN':'Venezuela','MWI':'Malawi','UKR':'Ukraine','TJK':'Tajikistan',
    'NIC':'Nicaragua','MRT':'Mauritania','SDN':'Sudan','GHA':'Ghana','ZMB':'Zambia',
    'ETH':'Ethiopia','PAK':'Pakistan','LKA':'Sri Lanka','LBN':'Lebanon','THA':'Thailand',
    'MYS':'Malaysia','PHL':'Philippines','ECU':'Ecuador','SLV':'El Salvador','ZWE':'Zimbabwe',
    'CHL':'Chile','COL':'Colombia','PER':'Peru','URY':'Uruguay','KAZ':'Kazakhstan',
}


def main():
    f_imp = to_html(fig_importance(), include_plotlyjs='cdn', full_html=False)
    f_mod = to_html(fig_models(), include_plotlyjs=False, full_html=False)
    f_yr = to_html(fig_peryear(), include_plotlyjs=False, full_html=False)

    pred, rank_map, n = load_country_data()
    country_figs = build_country_figs(pred)
    cinfo = json.loads((ROOT / "country_info.json").read_text())

    ranked = sorted(rank_map.items(), key=lambda kv: kv[1]['rank'])
    options = "".join(f'<option value="{iso}">{CNAMES.get(iso, iso)} ({iso})</option>'
                      for iso, _ in ranked)
    rank_json = json.dumps({iso: {'rank': v['rank'], 'pct': round(v['pct_rank']),
                                  'cal': round(100 * v['risk_calibrated'], 1),
                                  'last': v['last_date'], 'name': CNAMES.get(iso, iso),
                                  'blurb': cinfo.get(iso, {}).get('blurb', ''),
                                  'years': cinfo.get(iso, {}).get('crisis_years', [])}
                            for iso, v in rank_map.items()})
    tl_divs = "".join(f'<div class="tl" id="wrap_{iso}" style="display:none">{h}</div>'
                      for iso, h in country_figs.items())

    def t(term, definition):
        return f'<span class="term" tabindex="0">{term}<span class="tip">{definition}</span></span>'

    intro = (
        "<b>What is this?</b><br>"
        "Imagine you wake up and your country's money suddenly buys only half as much as "
        "yesterday. Savings shrink, prices jump, imports get expensive. That's a "
        + t("currency crisis", "When a country's money rapidly loses a big chunk of its value against the US dollar.")
        + ".<br><br>"
        "This project asks: <b>can we spot one coming about six months before it hits?</b> "
        "The model watches two kinds of warning signs " + DASH + " "
        + t("ordinary clues", "Normal signals: how much the currency has been slipping, and how bumpy it's been.")
        + " (everyday economic signals) and "
        + t("extreme-move clues", "Signals about rare, violent swings " + DASH + " the 'black swan' idea.")
        + " (rare, violent swings " + DASH + " the \"black swan\" idea from Nassim Taleb).<br><br>"
        "<b>The honest headlines:</b> crises are genuinely hard to predict; the extreme-move "
        "clues give a small but reliable improvement; and, surprisingly, they carry <b>most</b> "
        "of the useful warning signal. This is a research tool, not a crystal ball " + DASH +
        " it estimates <i>relative</i> risk (who's more at-risk than whom), not exact odds."
    )

    explain = (
        "<b>What are these three models, and what's going on?</b>"
        "<ul>"
        "<li><b>Simple model = Logistic Regression.</b> A straight-line method: it finds the "
        "best simple relationship between the clues and crisis risk. Easy to understand, but "
        "can only see simple patterns.</li>"
        "<li><b>Medium model = Random Forest.</b> Builds hundreds of little decision trees and "
        "averages them, so it can catch more twisty, non-straight-line patterns.</li>"
        "<li><b>Complex model = XGBoost.</b> A powerful, state-of-the-art method that builds "
        "trees one after another, each fixing the last one's mistakes. Usually the strongest "
        "tool " + DASH + " <i>when there's enough data.</i></li>"
        "</ul>"
        "<b>The interesting twist:</b> the extreme-move clues helped the <i>simple</i> model "
        "most (+33%) and the fancier models less (+7%). Why? The fancier models can partly "
        "<i>figure out</i> the extreme patterns on their own, so being handed them adds less. "
        "And strikingly, the <b>complex model didn't win overall</b> " + DASH + " because crises "
        "are so rare, even a powerful model can't find enough examples to learn from. That's a "
        "core lesson of this project: for rare, extreme events, <b>a simple model plus the right "
        "extreme-move clues beats brute-force complexity.</b> Fancier isn't automatically better."
    )

    glossary = (
        "<h2>Mini-glossary (plain English)</h2><table class='gloss'>"
        "<tr><td><b>Currency crisis</b></td><td>When a country's money rapidly loses a big chunk "
        "of its value against the US dollar (roughly a 30%+ drop).</td></tr>"
        "<tr><td><b>Ordinary clues</b></td><td>Normal signals " + DASH + " how much the currency "
        "has slipped lately and how bumpy it's been.</td></tr>"
        "<tr><td><b>Extreme-move clues</b></td><td>Signals about rare, violent swings: lopsidedness "
        "toward crashes, fat tails, counts of big sudden drops.</td></tr>"
        "<tr><td><b>Risk ranking</b></td><td>Where a country sits versus the others, from most "
        "at-risk (#1) to least. The model is good at <i>ordering</i> risk.</td></tr>"
        "<tr><td><b>Estimated risk (%)</b></td><td>The model's honest guess of crisis chance. It "
        "stays low even for risky countries " + DASH + " because these events are genuinely rare.</td></tr>"
        "<tr><td><b>Crisis window</b></td><td>The months just before a real crisis " + DASH + " what "
        "the model tries to flag in advance (recorded up to 2016).</td></tr></table>"
    )

    style = """
 body{font-family:Helvetica,Arial,sans-serif;max-width:1100px;margin:20px auto;
   color:#222;padding:0 16px;line-height:1.55}
 h1{color:#d1495b;margin-bottom:2px} h2{color:#3a6ea5;margin-top:38px}
 .intro{background:#f6f7f9;border-left:4px solid #d1495b;padding:14px 20px;border-radius:6px}
 .card{border:1px solid #e5e7eb;border-radius:8px;padding:8px;margin:12px 0}
 .how{background:#eef4fb;border-radius:6px;padding:8px 12px;font-size:.92em;color:#2b4a6b;margin-top:6px}
 .explain{background:#fff8f0;border:1px solid #f0e0cf;border-radius:8px;padding:14px 18px;margin-top:10px;font-size:.95em}
 .explain ul{margin:8px 0}
 .headline{font-size:1.3em;margin:12px 0 4px} .big{font-size:2.1em;color:#d1495b;font-weight:bold}
 .cinfo{background:#fbf7f2;border-radius:6px;padding:10px 14px;margin:8px 0;font-size:.95em}
 select{font-size:1em;padding:6px;border-radius:6px;border:1px solid #ccc}
 .note{color:#666;font-size:.9em}
 .term{border-bottom:1px dotted #d1495b;cursor:help;position:relative;color:#b03047;font-weight:bold}
 .term .tip{visibility:hidden;opacity:0;transition:opacity .15s;position:absolute;
   left:0;top:1.5em;z-index:10;background:#222;color:#fff;padding:8px 10px;border-radius:6px;
   width:260px;font-weight:normal;font-size:.85em;line-height:1.4}
 .term:hover .tip,.term:focus .tip{visibility:visible;opacity:1}
 table.gloss{border-collapse:collapse;width:100%}
 table.gloss td{border-top:1px solid #eee;padding:8px 10px;vertical-align:top}
 table.gloss td:first-child{width:200px;color:#3a6ea5}
"""

    js = ("const RANK = " + rank_json + "; const N = " + str(n) + ";\n"
          "function show(){\n"
          "  const iso = document.getElementById('picker').value;\n"
          "  document.querySelectorAll('.tl').forEach(d=>d.style.display='none');\n"
          "  const w = document.getElementById('wrap_'+iso); if(w) w.style.display='block';\n"
          "  const r = RANK[iso];\n"
          "  document.getElementById('rankline').innerHTML =\n"
          "    r.name + ' " + DASH + " risk ranking: <span class=\"big\">#'+r.rank+'</span> of '+N+\n"
          "    ' &nbsp;<span class=\"note\">(more at-risk than '+r.pct+'% of countries)</span><br>'+\n"
          "    '<span class=\"note\">Estimated risk as of '+r.last+': '+r.cal+'% " + DASH + " low in absolute terms, '+\n"
          "    'as expected for rare events. (2017 onward = prediction, not graded.)</span>';\n"
          "  let info = '';\n"
          "  if(r.blurb) info += r.blurb + '<br><br>';\n"
          "  if(r.years && r.years.length) info += '<b>Recorded currency-crisis years (Laeven" + DASH + "Valencia):</b> ' + r.years.join(', ') + '.';\n"
          "  else info += '<span class=\"note\">No crisis years recorded in the dataset for this country.</span>';\n"
          "  document.getElementById('cinfo').innerHTML = info;\n"
          "}\n"
          "window.onload = show;")

    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Beyond the Bell Curve</title><style>" + style + "</style></head><body>"
        "<h1>Beyond the Bell Curve</h1>"
        "<p class='note'>Can we see a currency crisis coming six months ahead? A research "
        "project across 110 emerging-market economies. Trained on 1990" + DASH + "2016 data, then "
        "applied to fresh data through <b>2023</b>. <b>Tip:</b> hover the "
        + t("underlined terms", "Like this! Hover any dotted term for a definition.") + ".</p>"
        "<div class='intro'>" + intro + "</div>"
        "<h2>1. Which warning signs matter most?</h2>"
        "<div class='card'>" + f_imp + "</div>"
        "<div class='how'><b>How to read this:</b> longer bar = the model relies on that warning "
        "sign more. The red bars (extreme-move clues) do most of the work " + DASH + " the project's "
        "central finding.</div>"
        "<h2>2. Do the extreme-move clues actually help?</h2>"
        "<div class='card'>" + f_mod + "</div>"
        "<div class='how'><b>How to read this:</b> each bar is one type of model. A positive bar "
        "means adding the extreme-move clues made it better at catching crises. "
        "<i>(Hover a bar for the real model name and scores.)</i></div>"
        "<div class='explain'>" + explain + "</div>"
        "<h2>3. Was that a fluke, or reliable?</h2>"
        "<div class='card'>" + f_yr + "</div>"
        "<div class='how'><b>How to read this:</b> one bar per year. Red = the extreme-move clues "
        "helped that year. They helped in most years " + DASH + " so it's a real pattern, not luck.</div>"
        "<h2>4. Explore a country (through 2023)</h2>"
        "<p>Pick a country: you'll see a short note on its currency history (to check the model "
        "against reality), how at-risk it ranks now, and its risk over time.<br><b>Note:</b> the "
        "model was trained only on 1990" + DASH + "2016 data. Values from 2017 onward are genuine "
        "<i>predictions on fresh data</i> " + DASH + " there's no official crisis list for those "
        "years to grade them, so they show who looks fragile, not verified accuracy.</p>"
        "<div class='card'><label>Country: </label>"
        "<select id='picker' onchange='show()'>" + options + "</select>"
        "<div class='headline' id='rankline'></div>"
        "<div class='cinfo' id='cinfo'></div>" + tl_divs + "</div>"
        + glossary +
        "<p class='note' style='margin-top:30px'>Built as an independent research project. The "
        "emphasis is honesty over hype " + DASH + " including reporting what the data <i>cannot</i> "
        "show.</p>"
        "<script>" + js + "</script></body></html>"
    )

    (OUT / "dashboard.html").write_text(html, encoding='utf-8')
    print(f"Saved: {OUT / 'dashboard.html'}")
    print("Open with:  open outputs/dashboard.html")


if __name__ == "__main__":
    main()
