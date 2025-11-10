import os
from pathlib import Path
from io import StringIO

from django.shortcuts import render, redirect
from django.conf import settings

import pandas as pd
import arff
from sklearn.model_selection import train_test_split

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from django.utils.text import slugify

# --- paths ---
MEDIA_ROOT = Path(settings.MEDIA_ROOT) if hasattr(settings, "MEDIA_ROOT") else Path("media")
UPLOADS_DIR = MEDIA_ROOT / "uploads"
SPLITS_DIR = MEDIA_ROOT / "splits"
PLOTS_DIR = MEDIA_ROOT / "plots"

for d in (UPLOADS_DIR, SPLITS_DIR, PLOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# --- helpers ---
def load_arff(path):
    with open(path, "r") as f:
        dataset = arff.load(f)
        attributes = [a[0] for a in dataset["attributes"]]
        return pd.DataFrame(dataset["data"], columns=attributes)


def df_info_text(df: pd.DataFrame) -> str:
    buf = StringIO()
    df.info(buf=buf)
    return buf.getvalue()


def save_dataframe_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)


def make_histogram(df: pd.DataFrame, base_slug: str) -> str:
    """Return relative path to saved histogram PNG."""
    num_cols = df.select_dtypes(include='number').columns.tolist()
    col = num_cols[0] if num_cols else df.columns[0]

    plt.figure(figsize=(6, 3.5))
    sns.histplot(df[col].dropna(), kde=True, color="#0ea5ff")
    plt.title(f"Distribución de {col}", color="white")
    plt.tight_layout()
    p = PLOTS_DIR / f"{base_slug}_hist.png"
    plt.savefig(p, dpi=80, bbox_inches="tight", facecolor="#0b0b0c")
    plt.close()
    return str(p.relative_to(MEDIA_ROOT))


def make_heatmap_train(train_df: pd.DataFrame, base_slug: str) -> str:
    df_num = train_df.select_dtypes(include='number')
    if df_num.shape[1] < 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No hay suficientes columnas numéricas para correlación",
                ha='center', va='center', color='white')
        ax.axis('off')
        p = PLOTS_DIR / f"{base_slug}_heatmap.png"
        fig.savefig(p, dpi=80, bbox_inches="tight", facecolor="#0b0b0c")
        plt.close(fig)
        return str(p.relative_to(MEDIA_ROOT))

    corr = df_num.corr()
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=False, cmap='Blues', cbar=True)
    plt.title("Matriz de correlación (train)", color="white")
    plt.tight_layout()
    p = PLOTS_DIR / f"{base_slug}_heatmap.png"
    plt.savefig(p, dpi=80, bbox_inches="tight", facecolor="#0b0b0c")
    plt.close()
    return str(p.relative_to(MEDIA_ROOT))


# --- views ---
def upload_dataset(request):
    context = {
        "head": None,
        "hist_path": None,
        "show_split": False,
    }

    if request.method == "POST" and request.FILES.get("dataset"):
        file = request.FILES["dataset"]
        filename = file.name
        lname = filename.lower()
        base_slug = slugify(Path(filename).stem) or "dataset"

        # save uploaded file
        save_path = UPLOADS_DIR / filename
        with open(save_path, "wb+") as dest:
            for chunk in file.chunks():
                dest.write(chunk)

        # load df
        try:
            if lname.endswith(".csv"):
                df = pd.read_csv(save_path)
            elif lname.endswith((".xlsx", ".xls")):
                df = pd.read_excel(save_path)
            elif lname.endswith(".arff"):
                df = load_arff(save_path)
            else:
                context["info"] = "Formato no soportado (CSV, XLSX o ARFF)."
                return render(request, "uploader/upload.html", context)
        except Exception as e:
            context["info"] = f"Error leyendo archivo: {e}"
            return render(request, "uploader/upload.html", context)

        # prepare outputs
        context["info"] = df_info_text(df)
        # Elegir solo las primeras 6 columnas para la vista previa (o las que quieras)
        preview_cols = df.columns[:6]  # ajusta el número según convenga
        preview_df = df[preview_cols].head(10)

        # convertir a HTML
        context["head"] = preview_df.to_html(classes="nice-table", index=False)
        hist_rel = make_histogram(df, base_slug)
        context["hist_path"] = f"{settings.MEDIA_URL}{hist_rel}"
        context["show_split"] = True
        


        # store info in session
        request.session["uploaded_filename"] = filename
        request.session["dataset_slug"] = base_slug

    return render(request, "uploader/upload.html", context)


def split_dataset(request):
    if request.method != "POST":
        return redirect("upload")

    filename = request.session.get("uploaded_filename")
    base_slug = request.session.get("dataset_slug")
    if not filename or not base_slug:
        return redirect("upload")

    save_path = UPLOADS_DIR / filename
    lname = filename.lower()
    try:
        if lname.endswith(".csv"):
            df = pd.read_csv(save_path)
        elif lname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(save_path)
        elif lname.endswith(".arff"):
            df = load_arff(save_path)
        else:
            return redirect("upload")
    except Exception:
        return redirect("upload")

    # get split options
    include_val = request.POST.get("include_val") == "on"
    save_splits = request.POST.get("save_splits") == "on"

    try:
        train_pct = float(request.POST.get("train_pct", 70))
        if include_val:
            val_pct = float(request.POST.get("val_pct", 10))
            test_pct = float(request.POST.get("test_pct", 20))
        else:
            val_pct = 0.0
            test_pct = float(request.POST.get("test_pct", 30))
    except Exception:
        return redirect("upload")

    # validate sum
    total_pct = train_pct + val_pct + test_pct
    if abs(total_pct - 100.0) > 1e-6:
        return render(request, "uploader/upload.html", {"info": "Los porcentajes deben sumar 100.", "show_split": True})

    # perform split
    if include_val:
        train_ratio = train_pct / 100.0
        rest_ratio = 1.0 - train_ratio
        val_ratio_within_rest = val_pct / (val_pct + test_pct) if (val_pct + test_pct) > 0 else 0.0
        train_df, rest_df = train_test_split(df, train_size=train_ratio, random_state=42, shuffle=True)
        if val_pct > 0:
            val_df, test_df = train_test_split(rest_df, test_size=(1 - val_ratio_within_rest), random_state=42, shuffle=True)
        else:
            val_df = pd.DataFrame()
            test_df = rest_df
    else:
        train_ratio = train_pct / 100.0
        train_df, test_df = train_test_split(df, train_size=train_ratio, random_state=42, shuffle=True)
        val_df = pd.DataFrame()

    # prepare context
    context = {
        "info": df_info_text(df),
        "train_preview": train_df.head().to_html(classes="table table-sm", index=False),
        "test_preview": test_df.head().to_html(classes="table table-sm", index=False),
        "val_preview": val_df.head().to_html(classes="table table-sm", index=False) if not val_df.empty else None,
        "splits_info": {
            "total_rows": len(df),
            "train_rows": len(train_df),
            "val_rows": len(val_df) if not val_df.empty else 0,
            "test_rows": len(test_df),
        },
        "hist_path": None,
        "heatmap_path": None,
        "saved_msg": None,
    }

    # generate heatmap
    heat_rel = make_heatmap_train(train_df, base_slug)
    context["heatmap_path"] = f"{settings.MEDIA_URL}{heat_rel}"

    # keep histogram
    hist_file = PLOTS_DIR / f"{base_slug}_hist.png"
    if hist_file.exists():
        context["hist_path"] = f"{settings.MEDIA_URL}{hist_file.relative_to(MEDIA_ROOT)}"

    # optionally save splits as CSV
    if save_splits:
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            save_dataframe_csv(train_df, SPLITS_DIR / f"{base_slug}_train.csv")
            save_dataframe_csv(test_df, SPLITS_DIR / f"{base_slug}_test.csv")
            if not val_df.empty:
                save_dataframe_csv(val_df, SPLITS_DIR / f"{base_slug}_val.csv")
            context["saved_msg"] = "Archivos guardados en media/splits/ (CSV)"
        except Exception as e:
            context["saved_msg"] = f"Error guardando splits: {e}"

    return render(request, "uploader/upload.html", context)
