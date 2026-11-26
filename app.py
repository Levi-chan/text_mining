from __future__ import annotations

import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import spacy
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(
    page_title="Eksploracja tekstu · Wesele",
    page_icon="📖",
    layout="wide",
)

CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}
WESELE_EPUB_PATH = Path("wesele.epub")
ANPWR_XLSX_PATH = Path("anpwr.xlsx")


@st.cache_resource(show_spinner="Ładowanie modelu języka polskiego (spaCy)...")
def load_nlp() -> spacy.Language:
    """Load Polish spaCy pipeline, downloading it if missing."""
    try:
        return spacy.load("pl_core_news_sm", disable=["parser", "ner"])
    except OSError:
        spacy.cli.download("pl_core_news_sm")
        return spacy.load("pl_core_news_sm", disable=["parser", "ner"])


@st.cache_data(show_spinner="Wczytywanie słownika ANPW_R...")
def load_lexicon(xlsx_path: str, neg_cut: float, pos_cut: float) -> pd.DataFrame:
    """Load ANPW_R affective norms dictionary and compute valence thresholds."""
    raw = pd.read_excel(
        xlsx_path,
        usecols=["polish word", "Valence_M", "arousal_M", "dominance_M"],
    )
    table = raw.rename(
        columns={
            "polish word": "word",
            "Valence_M": "valence",
            "arousal_M": "arousal",
            "dominance_M": "dominance",
        }
    ).copy()
    table["word"] = table["word"].astype(str).str.strip().str.lower()
    table = (
        table.dropna(subset=["word", "valence"])
        .drop_duplicates("word", keep="first")
        .reset_index(drop=True)
    )
    table["positive"] = table["valence"] >= pos_cut
    table["negative"] = table["valence"] <= neg_cut
    return table


def parse_wesele_epub(source: str | bytes, keep_didaskalia: bool = False) -> pd.DataFrame:
    """Extract speeches (and optionally stage directions) from Acts I-III of Wolne Lektury EPUB."""
    act_files = ["EPUB/part2.xhtml", "EPUB/part3.xhtml", "EPUB/part4.xhtml"]
    records: list[dict[str, str]] = []
    opener = io.BytesIO(source) if isinstance(source, bytes) else source

    with zipfile.ZipFile(opener) as archive:
        names = set(archive.namelist())
        if not all(part in names for part in act_files):
            return parse_generic_epub(archive)

        for part in act_files:
            soup = BeautifulSoup(archive.read(part), "lxml-xml")
            for anchor in soup.select("a.anchor"):
                anchor.decompose()

            book_root = soup.select_one("#book-text")
            if book_root is None:
                continue

            current_act = ""
            current_scene = ""
            current_speaker = ""

            for node in book_root.find_all(["h2", "div"], recursive=False):
                classes = node.get("class", [])
                if node.name == "h2":
                    text = node.get_text(" ", strip=True)
                    if "h2" in classes:
                        current_act = text
                    elif "h3" in classes:
                        current_scene = text
                    elif "h4" in classes:
                        current_speaker = text
                elif node.name == "div":
                    if "didaskalia" in classes and keep_didaskalia:
                        text = node.get_text(" ", strip=True)
                        if text:
                            records.append(
                                {
                                    "act": current_act or "Tekst",
                                    "scene": current_scene,
                                    "speaker": "[DIDASKALIA]",
                                    "kind": "didaskalia",
                                    "text": text,
                                }
                            )
                    elif "kwestia" in classes:
                        if not keep_didaskalia:
                            for nested in node.select(".didaskalia"):
                                nested.decompose()
                        verses = [v.get_text(" ", strip=True) for v in node.select("div.verse")]
                        text = " ".join(v for v in verses if v)
                        if text:
                            records.append(
                                {
                                    "act": current_act or "Tekst",
                                    "scene": current_scene,
                                    "speaker": current_speaker or "Nieznany",
                                    "kind": "kwestia",
                                    "text": text,
                                }
                            )

    return pd.DataFrame(records)


def parse_generic_epub(archive: zipfile.ZipFile) -> pd.DataFrame:
    """Fallback parser for arbitrary EPUB archives."""
    xhtml_files = sorted(
        name for name in archive.namelist() if name.lower().endswith((".xhtml", ".html", ".htm"))
    )
    records: list[dict[str, str]] = []
    for idx, name in enumerate(xhtml_files, start=1):
        soup = BeautifulSoup(archive.read(name), "lxml")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all(["p", "div"]) if not p.find(["p", "div"])]
        text = " ".join(p for p in paragraphs if p)
        if text:
            records.append(
                {
                    "act": f"Część {idx}",
                    "scene": name,
                    "speaker": "Tekst",
                    "kind": "tekst",
                    "text": text,
                }
            )
    return pd.DataFrame(records)


def parse_plain_text(raw_text: str) -> pd.DataFrame:
    """Split plain text into non-empty paragraphs for exploration."""
    paragraphs = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not paragraphs:
        return pd.DataFrame(columns=["act", "scene", "speaker", "kind", "text"])
    return pd.DataFrame(
        [
            {
                "act": "Tekst",
                "scene": f"Akapit {i}",
                "speaker": "Tekst",
                "kind": "tekst",
                "text": paragraph,
            }
            for i, paragraph in enumerate(paragraphs, start=1)
        ]
    )


@st.cache_data(show_spinner="Lematyzacja i dopasowanie słownika...")
def analyze_speeches(
    speeches_df: pd.DataFrame,
    lexicon_df: pd.DataFrame,
    content_words_only: bool,
) -> pd.DataFrame:
    """Lemmatize text with spaCy and merge with the ANPW_R lexicon."""
    nlp = load_nlp()
    token_rows: list[dict[str, object]] = []
    pos_idx = 0

    for speech_id, (row, doc) in enumerate(
        zip(speeches_df.itertuples(index=False), nlp.pipe(speeches_df["text"].tolist(), batch_size=64))
    ):
        for tok in doc:
            if not tok.is_alpha:
                continue
            if content_words_only and tok.pos_ not in CONTENT_POS:
                continue
            token_rows.append(
                {
                    "position": pos_idx,
                    "speech_id": speech_id,
                    "act": row.act,
                    "scene": row.scene,
                    "speaker": row.speaker,
                    "surface": tok.text,
                    "lemma": tok.lemma_.lower(),
                    "pos": tok.pos_,
                }
            )
            pos_idx += 1

    if not token_rows:
        return pd.DataFrame(
            columns=[
                "position",
                "speech_id",
                "act",
                "scene",
                "speaker",
                "surface",
                "lemma",
                "pos",
                "word",
                "valence",
                "arousal",
                "positive",
                "negative",
                "matched",
            ]
        )

    tokens_df = pd.DataFrame(token_rows).merge(
        lexicon_df[["word", "valence", "arousal", "positive", "negative"]],
        left_on="lemma",
        right_on="word",
        how="left",
    )
    tokens_df["positive"] = tokens_df["positive"].fillna(False).astype(bool)
    tokens_df["negative"] = tokens_df["negative"].fillna(False).astype(bool)
    tokens_df["matched"] = tokens_df["valence"].notna()
    return tokens_df


def build_trajectory(tokens_df: pd.DataFrame, section_size: int) -> pd.DataFrame:
    """Aggregate token-level sentiment metrics into fixed-size word sections."""
    df = tokens_df.copy()
    df["section"] = (df["position"] // section_size).astype(int)
    trajectory = df.groupby("section", as_index=False).agg(
        total=("lemma", "size"),
        matched=("matched", "sum"),
        positive=("positive", "sum"),
        negative=("negative", "sum"),
        mean_valence=("valence", "mean"),
        act=("act", lambda s: s.mode().iloc[0]),
    )
    trajectory["net"] = trajectory["positive"] - trajectory["negative"]
    trajectory["net_per_1000"] = 1000 * trajectory["net"] / trajectory["total"]
    trajectory["coverage_percent"] = 100 * trajectory["matched"] / trajectory["total"]
    return trajectory


def plot_trajectory(
    trajectory: pd.DataFrame,
    metric: str,
    rolling_window: int,
    section_size: int,
) -> plt.Figure:
    """Create sentiment trajectory and dictionary coverage chart."""
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    y_label = "Średnia walencja (1–9)" if metric == "mean_valence" else "Bilans na 1000 słów"
    ref_value = 5.0 if metric == "mean_valence" else 0.0
    smooth = trajectory[metric].rolling(window=max(1, rolling_window), center=True, min_periods=1).mean()

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(11, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1]},
        layout="constrained",
    )
    ax_top.plot(trajectory["section"], trajectory[metric], color="#93b1d1", linewidth=1.1, label="Wartość w sekcji")
    if rolling_window > 1:
        ax_top.plot(
            trajectory["section"],
            smooth,
            color="#235789",
            linewidth=2.0,
            label=f"Średnia krocząca ({rolling_window})",
        )
    ax_top.axhline(ref_value, color="#b54b53", linestyle="--", linewidth=0.9, label=f"Poziom odniesienia ({ref_value})")

    ax_top.margins(y=0.14)
    act_changes = trajectory.loc[trajectory["act"] != trajectory["act"].shift(1), ["section", "act"]]
    y_top = ax_top.get_ylim()[1]
    for row in act_changes.itertuples():
        if row.section > 0:
            ax_top.axvline(row.section - 0.5, color="#6b7280", linestyle=":", linewidth=1.0)
            ax_bot.axvline(row.section - 0.5, color="#6b7280", linestyle=":", linewidth=1.0)
        ax_top.text(
            row.section + 0.3,
            y_top * 0.995,
            str(row.act),
            va="top",
            fontsize=9,
            color="#374151",
            fontweight="bold",
        )

    ax_top.set(title=f"Przebieg w blokach po {section_size} słów", ylabel=y_label)
    ax_top.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.85, edgecolor="none", ncol=3)

    ax_bot.bar(trajectory["section"], trajectory["coverage_percent"], color="#6488a8", alpha=0.85, width=0.8)
    ax_bot.set(ylabel="Pokrycie (%)", xlabel=f"Numer sekcji (po {section_size} słów)")
    return fig


def main() -> None:
    st.title("Eksploracja tekstu · Analiza sentymentu i walencji")
    st.caption("Lematyzacja `spaCy` (`pl_core_news_sm`) + słownik afektywny **ANPW_R** (Imbir 2016)")

    if not ANPWR_XLSX_PATH.is_file():
        st.error("Brak pliku `anpwr.xlsx` w katalogu projektu.")
        st.stop()

    with st.sidebar:
        st.header("Źródło tekstu")
        source_mode = st.radio(
            "Wybierz dane wejściowe",
            ["Wesele (wesele.epub)", "Wgraj własny plik (.epub / .txt)", "Wklej własny tekst"],
        )

        include_didaskalia = False
        uploaded_file = None
        pasted_text = ""

        if source_mode == "Wesele (wesele.epub)":
            include_didaskalia = st.checkbox("Uwzględnij didaskalia", value=False)
        elif source_mode == "Wgraj własny plik (.epub / .txt)":
            uploaded_file = st.file_uploader("Plik EPUB lub TXT", type=["epub", "txt"])
        else:
            pasted_text = st.text_area("Wklej tekst do analizy", height=180)

        st.divider()
        st.header("Parametry analizy")
        content_words_only = st.checkbox(
            "Tylko słowa pełnoznaczne (NOUN, VERB, ADJ, ADV)",
            value=False,
        )
        section_size = st.slider("Rozmiar sekcji (liczba słów)", min_value=50, max_value=2000, value=300, step=50)
        rolling_window = st.slider("Okno średniej kroczącej", min_value=1, max_value=15, value=3, step=1)
        metric_label = st.selectbox(
            "Metryka wykresu",
            options=["Średnia walencja (1–9)", "Bilans na 1000 słów"],
        )
        metric = "mean_valence" if metric_label.startswith("Średnia") else "net_per_1000"

        col_neg, col_pos = st.columns(2)
        with col_neg:
            neg_cut = st.number_input("Próg negatywny <=", min_value=1.0, max_value=5.0, value=4.0, step=0.2)
        with col_pos:
            pos_cut = st.number_input("Próg pozytywny >=", min_value=5.0, max_value=9.0, value=6.0, step=0.2)

    if source_mode == "Wesele (wesele.epub)":
        if not WESELE_EPUB_PATH.is_file():
            st.error("Nie znaleziono pliku `wesele.epub` w katalogu projektu.")
            st.stop()
        speeches_df = parse_wesele_epub(str(WESELE_EPUB_PATH), keep_didaskalia=include_didaskalia)
    elif source_mode == "Wgraj własny plik (.epub / .txt)":
        if uploaded_file is None:
            st.info("Wgraj plik `.epub` lub `.txt` w panelu po lewej stronie.")
            st.stop()
        raw_bytes = uploaded_file.getvalue()
        if uploaded_file.name.lower().endswith(".epub"):
            speeches_df = parse_wesele_epub(raw_bytes, keep_didaskalia=True)
        else:
            speeches_df = parse_plain_text(raw_bytes.decode("utf-8", errors="replace"))
    else:
        if not pasted_text.strip():
            st.info("Wklej tekst w panelu po lewej stronie, aby uruchomić analizę.")
            st.stop()
        speeches_df = parse_plain_text(pasted_text)

    if speeches_df.empty:
        st.warning("Nie udało się wyodrębnić fragmentów tekstu z wybranego źródła.")
        st.stop()

    lexicon_df = load_lexicon(str(ANPWR_XLSX_PATH), neg_cut, pos_cut)
    tokens_df = analyze_speeches(speeches_df, lexicon_df, content_words_only)

    if tokens_df.empty:
        st.warning("Brak tokenów do analizy po zastosowaniu bieżących filtrów.")
        st.stop()

    trajectory = build_trajectory(tokens_df, section_size)
    tokens_df = tokens_df.assign(section=(tokens_df["position"] // section_size).astype(int))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fragmenty", f"{len(speeches_df):,}".replace(",", " "))
    c2.metric("Tokeny", f"{len(tokens_df):,}".replace(",", " "))
    c3.metric("Pokrycie słownika", f"{100 * tokens_df['matched'].mean():.1f}%")
    mean_val = tokens_df["valence"].mean()
    c4.metric("Średnia walencja", f"{mean_val:.2f}" if pd.notna(mean_val) else "—")

    tab_trend, tab_section, tab_speakers = st.tabs(
        ["Przebieg sentymentu", "Inspekcja sekcji", "Postacie i słownictwo"]
    )

    with tab_trend:
        fig = plot_trajectory(trajectory, metric, rolling_window, section_size)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.subheader("Podsumowanie według części / aktów")
        summary_df = (
            trajectory.groupby("act", sort=False)[
                ["total", "matched", "coverage_percent", "mean_valence", "net_per_1000"]
            ]
            .mean()
            .round(2)
            .rename(
                columns={
                    "total": "Śr. słów w sekcji",
                    "matched": "Śr. dopasowanych",
                    "coverage_percent": "Pokrycie (%)",
                    "mean_valence": "Śr. walencja",
                    "net_per_1000": "Bilans / 1000 słów",
                }
            )
        )
        st.dataframe(summary_df, use_container_width=True)

    with tab_section:
        full_sections = trajectory.loc[trajectory["total"] == section_size]
        if full_sections.empty:
            full_sections = trajectory

        preset = st.radio(
            "Wybór sekcji",
            ["Najniższa wartość", "Najwyższa wartość", "Ręcznie"],
            horizontal=True,
        )
        if preset == "Najniższa wartość" and full_sections[metric].notna().any():
            section_number = int(full_sections.loc[full_sections[metric].idxmin(), "section"])
        elif preset == "Najwyższa wartość" and full_sections[metric].notna().any():
            section_number = int(full_sections.loc[full_sections[metric].idxmax(), "section"])
        else:
            section_number = st.slider(
                "Numer sekcji",
                min_value=0,
                max_value=int(trajectory["section"].max()),
                value=0,
            )

        selected_row = trajectory.loc[trajectory["section"] == section_number]
        sec_tokens = tokens_df.loc[tokens_df["section"] == section_number]

        st.markdown(f"**Sekcja {section_number}** · {selected_row['act'].iloc[0]}")
        st.dataframe(
            selected_row[
                ["total", "matched", "coverage_percent", "mean_valence", "positive", "negative", "net_per_1000"]
            ].round(2),
            use_container_width=True,
            hide_index=True,
        )

        matched_words = (
            sec_tokens.loc[sec_tokens["matched"]]
            .groupby("lemma", as_index=False)
            .agg(
                Liczba=("lemma", "size"),
                Walencja=("valence", "first"),
                Pobudzenie=("arousal", "first"),
                Formy=("surface", lambda s: ", ".join(pd.unique(s)[:3])),
            )
            .sort_values(["Walencja", "Liczba"], ascending=[True, False])
            .round(2)
        )
        st.subheader("Dopasowane słowa ze słownika ANPW_R")
        st.dataframe(matched_words, use_container_width=True, hide_index=True)

        st.subheader("Oryginalne fragmenty tekstu")
        speech_ids = sec_tokens["speech_id"].unique()
        for row in speeches_df.iloc[speech_ids].itertuples():
            prefix = f"**[{row.scene}]** " if row.scene else ""
            st.markdown(f"{prefix}**{row.speaker}:** {row.text}")

    with tab_speakers:
        speaker_stats = (
            tokens_df.groupby("speaker", as_index=False)
            .agg(
                Słowa=("lemma", "size"),
                Dopasowane=("matched", "sum"),
                Walencja=("valence", "mean"),
                Pozytywne=("positive", "sum"),
                Negatywne=("negative", "sum"),
            )
            .sort_values("Słowa", ascending=False)
            .round(2)
        )
        st.subheader("Statystyki według mówcy / sekcji")
        st.dataframe(speaker_stats, use_container_width=True, hide_index=True)

        col_low, col_high = st.columns(2)
        matched_all = tokens_df.loc[tokens_df["matched"]]
        top_lemmas = (
            matched_all.groupby("lemma", as_index=False)
            .agg(Liczba=("lemma", "size"), Walencja=("valence", "first"))
        )
        with col_low:
            st.subheader("Najbardziej negatywne lematy")
            st.dataframe(
                top_lemmas.sort_values(["Walencja", "Liczba"], ascending=[True, False]).head(15).round(2),
                use_container_width=True,
                hide_index=True,
            )
        with col_high:
            st.subheader("Najbardziej pozytywne lematy")
            st.dataframe(
                top_lemmas.sort_values(["Walencja", "Liczba"], ascending=[False, False]).head(15).round(2),
                use_container_width=True,
                hide_index=True,
            )


if __name__ == "__main__":
    main()
