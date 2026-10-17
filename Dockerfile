FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/opt/lightning
RUN pip install --no-cache-dir numpy==2.5.4 pillow==13.0.0 && useradd --system --uid 10001 preflight
WORKDIR /opt/lightning
COPY preflight.py plugin.py /opt/lightning/Preflight_checker/
USER preflight
ENTRYPOINT ["python", "-m", "Preflight_checker.plugin"]
