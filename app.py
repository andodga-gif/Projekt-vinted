import streamlit as st
from rembg import remove, new_session
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw
import numpy as np
import io
import random
import math

st.set_page_config(page_title="Vinted Studio AI", page_icon="👕", layout="centered")

st.markdown("<h1 style='text-align: center;'>👕 Vinted Studio AI</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: gray;'>Wgraj zdjęcie ubrania, aby automatycznie "
    "poprawić tło i cienie bez utraty detali.</p>",
    unsafe_allow_html=True,
)


@st.cache_resource
def zaladuj_model():
    # isnet-general-use (~179 MB) zamiast bria-rmbg / RMBG-2.0 (~1,02 GB) —
    # ten drugi sam w sobie przekraczał limit RAM na darmowym Streamlit Cloud
    # (kontener padał już przy wczytywaniu modelu, zanim zdjęcie trafiało do kodu).
    return new_session("isnet-general-use")


sesja_ai = zaladuj_model()

# ----------------------------------------------------------------------------------
# 1. PROCEDURALNE TŁA (bez zewnętrznych plików graficznych – działa wszędzie, w tym
#    na darmowym Streamlit Cloud, bez potrzeby wgrywania dużych zdjęć tekstur).
# ----------------------------------------------------------------------------------

def _szum_wielooktawowy(w, h, skale=(4, 8, 16, 32, 64, 128), zanik=0.55, seed=None):
    """Sumuje kilka warstw wygładzonego szumu w różnych skalach -> wygląda jak
    naturalna, 'organiczna' tekstura (podobnie do szumu Perlina), a używa
    tylko numpy + PIL.resize (bez dodatkowych bibliotek)."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    wynik = np.zeros((h, w), dtype=np.float32)
    amplituda = 1.0
    suma_amplitud = 0.0
    for skala in skale:
        mw, mh = max(2, w // skala), max(2, h // skala)
        mala = rng.random((mh, mw)).astype(np.float32)
        warstwa = Image.fromarray((mala * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
        wynik += amplituda * (np.asarray(warstwa, dtype=np.float32) / 255.0)
        suma_amplitud += amplituda
        amplituda *= zanik
    wynik /= suma_amplitud
    return wynik  # wartości 0..1


def generuj_beton(w, h, seed=None):
    """Jasny, loftowy beton: baza szarości + mikroszczegóły + delikatne rysy/plamy
    + winieta, żeby wyglądało jak realne zdjęcie podłogi studyjnej."""
    baza = _szum_wielooktawowy(w, h, skale=(3, 6, 12, 24, 48, 96), seed=seed)
    # kontrast tekstury – lekko spłaszczamy skrajności, żeby nie było "hałaśliwie"
    baza = np.clip((baza - 0.5) * 0.9 + 0.5, 0, 1)

    # bazowy jasny odcień betonu z lekkim ciepłym/chłodnym driftem
    kolor_bazowy = np.array([214, 213, 210], dtype=np.float32)
    obraz = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(3):
        obraz[:, :, i] = kolor_bazowy[i] + (baza - 0.5) * 34.0

    # drobny szum ziarnisty (mikroteksturę betonu)
    ziarno = (np.random.default_rng(seed).normal(0, 4.5, (h, w, 1))).astype(np.float32)
    obraz += ziarno

    # kilka subtelnych "rys" / spękań jako cienkie, losowe linie
    rng = np.random.default_rng(seed)
    warstwa_rys = np.zeros((h, w), dtype=np.float32)
    for _ in range(rng.integers(3, 7)):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        dlugosc = rng.integers(int(w * 0.15), int(w * 0.5))
        kat = rng.uniform(0, math.pi)
        x1 = int(np.clip(x0 + dlugosc * math.cos(kat), 0, w - 1))
        y1 = int(np.clip(y0 + dlugosc * math.sin(kat), 0, h - 1))
        rysa_img = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(rysa_img)
        d.line([(x0, y0), (x1, y1)], fill=255, width=1)
        warstwa_rys += np.asarray(rysa_img.filter(ImageFilter.GaussianBlur(1.2)), dtype=np.float32) / 255.0
    for i in range(3):
        obraz[:, :, i] -= warstwa_rys * 6.0

    # winieta studyjna (delikatne przyciemnienie narożników)
    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    winieta = np.clip(1.0 - 0.18 * (x ** 2 + y ** 2), 0.75, 1.0)
    for i in range(3):
        obraz[:, :, i] *= winieta

    obraz = np.clip(obraz, 0, 255).astype(np.uint8)
    tlo = Image.fromarray(obraz, mode="RGB").convert("RGBA")
    tlo = tlo.filter(ImageFilter.GaussianBlur(0.4))  # lekkie zmiękczenie ziarna
    return tlo


def generuj_drewno(w, h, seed=None):
    """Ciepłe drewno loftowe: deski (pionowe pasy) + słoje symulowane sinusami
    + szum + winieta."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    X, Y = np.meshgrid(x, y)

    szerokosc_deski = rng.uniform(0.10, 0.16)
    numer_deski = np.floor(X / szerokosc_deski)

    # słoje drewna: sinusoidalne pasy zniekształcone szumem wielooktawowym
    zniekszt = (_szum_wielooktawowy(w, h, skale=(6, 12, 24, 48), seed=seed) - 0.5) * 0.06
    slouje = np.sin((X + zniekszt) / 0.01 * 2 * math.pi * 3) * 0.5 + 0.5
    slouje = slouje ** 3  # wyostrzenie linii słojów

    # każda deska ma nieco inny odcień
    odcienie = rng.uniform(-14, 14, size=int(numer_deski.max()) + 2)
    roznica_deski = odcienie[numer_deski.astype(int)]

    baza = np.array([176, 140, 104], dtype=np.float32)  # ciepły jasny dąb
    obraz = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(3):
        obraz[:, :, i] = baza[i] + roznica_deski + (slouje - 0.5) * 22.0

    # delikatna szczelina między deskami
    modulo = np.mod(X, szerokosc_deski)
    szczelina = (modulo < (0.9 / w)) | (modulo > szerokosc_deski - (0.9 / w))
    for i in range(3):
        obraz[:, :, i][szczelina] -= 30

    ziarno = rng.normal(0, 3.5, (h, w, 1)).astype(np.float32)
    obraz += ziarno

    xg, yg = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    winieta = np.clip(1.0 - 0.16 * (xg ** 2 + yg ** 2), 0.78, 1.0)
    for i in range(3):
        obraz[:, :, i] *= winieta

    obraz = np.clip(obraz, 0, 255).astype(np.uint8)
    tlo = Image.fromarray(obraz, mode="RGB").convert("RGBA")
    tlo = tlo.filter(ImageFilter.GaussianBlur(0.3))
    return tlo


# ----------------------------------------------------------------------------------
# 2. WYGŁADZANIE KRAWĘDZI (tylko maska alfa – nadruki i tekstura pozostają nietknięte)
# ----------------------------------------------------------------------------------

def wygladz_krawedzie(rgba, sila=1.0):
    """Wygładza poszarpane / pofalowane brzegi wycięcia, operując WYŁĄCZNIE na
    kanale alfa (masce). Piksele RGB (nadruk, szwy, logo) nie są w ogóle
    modyfikowane – zmienia się jedynie kontur przezroczystości.
    `sila` to mnożnik (1.0 = domyślnie), sterowany suwakiem w UI.

    Wydajność: duże filtry rangowe (Max/MinFilter) na zdjęciach 10+ Mpx są
    bardzo kosztowne. Dlatego operacje morfologiczne robimy na pomniejszonej
    kopii maski (max. 700 px krótszego boku) i skalujemy wynik z powrotem —
    efekt wygładzenia jest ten sam, ale dziesiątki razy szybszy."""
    w, h = rgba.size
    baza = min(w, h)
    r, g, b, a = rgba.split()

    robocza_baza = min(baza, 700)
    skala = robocza_baza / baza
    if skala < 1.0:
        robocze_wh = (max(1, round(w * skala)), max(1, round(h * skala)))
        a_robocza = a.resize(robocze_wh, Image.BILINEAR)
    else:
        a_robocza = a

    kernel = max(3, int(round(robocza_baza / 200 * sila)))
    if kernel % 2 == 0:
        kernel += 1
    promien_rozmycia = max(1.0, robocza_baza * 0.0025 * sila)

    # 1) zamknięcie morfologiczne (dylatacja + erozja) usuwa drobne "ząbki"
    a_robocza = a_robocza.filter(ImageFilter.MaxFilter(kernel))
    a_robocza = a_robocza.filter(ImageFilter.MinFilter(kernel))
    # 2) otwarcie morfologiczne usuwa drobne wypustki/pofalowania
    a_robocza = a_robocza.filter(ImageFilter.MinFilter(kernel))
    a_robocza = a_robocza.filter(ImageFilter.MaxFilter(kernel))
    # 3) lekkie rozmycie konturu
    a_robocza = a_robocza.filter(ImageFilter.GaussianBlur(promien_rozmycia))

    if skala < 1.0:
        a_final_img = a_robocza.resize((w, h), Image.LANCZOS)
    else:
        a_final_img = a_robocza

    a_np = np.asarray(a_final_img, dtype=np.float32)
    # delikatne podbicie kontrastu maski (NIE twardy próg) – kontur staje się
    # bardziej zdecydowany, ale naturalnie miękkie krawędzie (np. strzępiące
    # się nitki) nie zamieniają się w twardo wycięty kontur
    a_np = np.clip(128 + (a_np - 128) * 1.15, 0, 255)
    a_final = Image.fromarray(a_np.astype(np.uint8), mode="L")

    return Image.merge("RGBA", (r, g, b, a_final))


def wysrodkuj(rgba, margines_proc=0.06):
    """Automatyczne centrowanie: znajduje bounding box widocznej odzieży
    i umieszcza ją na środku płótna o tym samym rozmiarze, z równym marginesem."""
    alfa = np.asarray(rgba.split()[-1])
    ys, xs = np.where(alfa > 8)
    if len(xs) == 0:
        return rgba
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    wycinek = rgba.crop((x0, y0, x1 + 1, y1 + 1))

    w, h = rgba.size
    margines = int(min(w, h) * margines_proc)
    dostepna_w, dostepna_h = w - 2 * margines, h - 2 * margines
    skala = min(dostepna_w / wycinek.width, dostepna_h / wycinek.height, 1.0)
    if skala < 1.0:
        nowy_rozmiar = (max(1, int(wycinek.width * skala)), max(1, int(wycinek.height * skala)))
        wycinek = wycinek.resize(nowy_rozmiar, Image.LANCZOS)

    plotno = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    px = (w - wycinek.width) // 2
    py = (h - wycinek.height) // 2
    plotno.paste(wycinek, (px, py), wycinek)
    return plotno


# ----------------------------------------------------------------------------------
# 3. PODWÓJNY CIEŃ 3D: ostry cień kontaktowy + miękki cień otoczenia
# ----------------------------------------------------------------------------------

def generuj_cienie(rgba):
    w, h = rgba.size
    baza = min(w, h)  # punkt odniesienia do skalowania - działa dla każdej rozdzielczości
    maska = rgba.split()[-1]

    warstwa_cieni = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    # --- cień kontaktowy (ostry, ciemny, tuż pod ubraniem) ---
    promien_kontakt = max(3, baza * 0.010)
    offset_kontakt = (max(1, round(baza * 0.006)), max(1, round(baza * 0.010)))
    kontakt = Image.new("RGBA", (w, h), (10, 10, 12, 255))
    kontakt = Image.composite(kontakt, Image.new("RGBA", (w, h), (0, 0, 0, 0)), maska)
    kontakt = kontakt.filter(ImageFilter.GaussianBlur(radius=promien_kontakt))
    alfa_kontakt = kontakt.split()[-1].point(lambda p: int(p * 0.55))
    kontakt.putalpha(alfa_kontakt)
    warstwa_cieni.paste(kontakt, offset_kontakt, kontakt)

    # --- cień otoczenia (miękki, szeroki, dalej od ubrania) ---
    promien_otoczenie = max(10, baza * 0.045)
    offset_otoczenie = (max(2, round(baza * 0.018)), max(3, round(baza * 0.032)))
    otoczenie = Image.new("RGBA", (w, h), (20, 20, 26, 255))
    otoczenie = Image.composite(otoczenie, Image.new("RGBA", (w, h), (0, 0, 0, 0)), maska)
    otoczenie = otoczenie.filter(ImageFilter.GaussianBlur(radius=promien_otoczenie))
    alfa_otoczenie = otoczenie.split()[-1].point(lambda p: int(p * 0.32))
    otoczenie.putalpha(alfa_otoczenie)

    finalne = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    finalne.paste(otoczenie, offset_otoczenie, otoczenie)
    finalne = Image.alpha_composite(finalne, warstwa_cieni)
    return finalne


# ----------------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------------

typ_tla = st.radio("Wybierz styl tła:", ("Jasny Beton", "Ciepłe Drewno"), horizontal=True)
sila_wygladzania = st.slider(
    "Siła wygładzania krawędzi", min_value=0.5, max_value=3.0, value=1.0, step=0.25,
    help="Wyższa wartość mocniej prostuje pofalowane brzegi ubrania. Zbyt wysoka może zaokrąglić naturalne detale kroju (np. rozcięcia, kaptur)."
)
plik_foto = st.file_uploader("Wybierz zdjęcie z galerii lub zrób aparatem", type=["jpg", "jpeg", "png"])

if plik_foto is not None:
    img = Image.open(plik_foto).convert("RGBA")
    img = ImageOps.exif_transpose(img)

    # Ograniczenie rozdzielczości roboczej: zdjęcia z telefonu (12+ Mpx) zużywają
    # dużo RAM-u na darmowym, ograniczonym serwerze (model AI sam waży ~1 GB).
    # 2000 px dłuższego boku to więcej niż potrzeba do dobrej jakości na Vinted/
    # e-commerce (typowe wyświetlanie i tak skaluje w dół), a drastycznie
    # zmniejsza zużycie pamięci i czas przetwarzania.
    MAX_BOK = 2000
    if max(img.size) > MAX_BOK:
        skala = MAX_BOK / max(img.size)
        nowy_rozmiar = (round(img.width * skala), round(img.height * skala))
        img = img.resize(nowy_rozmiar, Image.LANCZOS)

    st.image(img, caption="Oryginalne zdjęcie", width='stretch')

    if st.button("✨ GENERUJ PRODUKTOWE FOTO ✨", type="primary", width='stretch'):
        with st.spinner("Przetwarzanie AI... Zachowuję oryginalne detale i logo."):

            # 1. Wycinanie starego tła modelem rmbg-2.0
            ubranie_czyste = remove(img, session=sesja_ai)

            # 2. Wygładzenie tylko konturu (nadruk/logo nienaruszone)
            ubranie_czyste = wygladz_krawedzie(ubranie_czyste, sila=sila_wygladzania)

            # 3. Automatyczne centrowanie
            ubranie_czyste = wysrodkuj(ubranie_czyste)

            # 4. Subtelne podbicie kontrastu tkaniny
            r, g, b, a = ubranie_czyste.split()
            rgb_only = Image.merge("RGB", (r, g, b))
            rgb_only = ImageEnhance.Contrast(rgb_only).enhance(1.04)
            rgb_only = ImageEnhance.Sharpness(rgb_only).enhance(1.08)
            ubranie_czyste = Image.merge("RGBA", (*rgb_only.split(), a))

            # 5. Podwójny cień 3D
            cien_final = generuj_cienie(ubranie_czyste)

            # 6. Realistyczne tło proceduralne
            w, h = img.size
            seed = random.randint(0, 999999)
            if typ_tla == "Jasny Beton":
                tlo = generuj_beton(w, h, seed=seed)
            else:
                tlo = generuj_drewno(w, h, seed=seed)

            # 7. Składanie warstw: tło -> cienie -> ubranie
            foto_koncowe = Image.alpha_composite(tlo, cien_final)
            foto_koncowe = Image.alpha_composite(foto_koncowe, ubranie_czyste)
            gotowy_obraz = foto_koncowe.convert("RGB")

            st.success("Gotowe!")
            st.image(gotowy_obraz, caption="Wynik końcowy", width='stretch')

            bufor = io.BytesIO()
            gotowy_obraz.save(bufor, format="JPEG", quality=100)
            bajt_obrazu = bufor.getvalue()

            st.download_button(
                label="📥 Pobierz gotowe zdjęcie",
                data=bajt_obrazu,
                file_name="vinted_studio.jpg",
                mime="image/jpeg",
                width='stretch',
            )
