FROM python:3.11-slim

# WeasyPrint's real dependencies (Pango/cairo/gdk-pixbuf) — this is what
# lets render_pdf() use WeasyPrint instead of the xhtml2pdf fallback that
# Windows dev boxes without these libs fall back to.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libcairo2 \
    libffi-dev shared-mime-info \
    ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY conftest.py .

RUN mkdir -p uploads

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
