# Eksploracja tekstu — Analiza sentymentu i walencji (*Wesele*)

Aplikacja webowa w **Streamlit** do eksploracji tekstu w języku polskim (lematyzacja `spaCy` + słownik afektywny **ANPW_R**).

## Zawartość repozytorium
* `app.py` — aplikacja Streamlit (analiza dramatu *Wesele* z `wesele.epub` oraz własnych plików `.txt` / `.epub`)
* `wesele_sentyment_v0_1.ipynb` — pierwotny prototyp analizy w notatniku Jupyter
* `wesele.epub` — tekst dramatu Stanisława Wyspiańskiego (Wolne Lektury)
* `anpwr.xlsx` — słownik *Affective Norms for 4900 Polish Words Reload* (Imbir 2016)
* `requirements.txt` — zależności do uruchomienia lokalnie oraz na Streamlit Community Cloud

## Uruchomienie lokalne (Windows)
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Deploy na Streamlit Community Cloud
1. Zaloguj się na [share.streamlit.io](https://share.streamlit.io) przez GitHuba.
2. Kliknij **Create app** → wybierz to repozytorium, gałąź `main` i plik `app.py`.
3. Kliknij **Deploy**.
