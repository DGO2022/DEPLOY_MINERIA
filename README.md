# 🧠 Demo ST — Streamlit
🤖 Predicción de Diabetes (Pima Dataset de `statsmodels.api.datasets.get_rdataset("Pima.tr", "MASS").data` ) para aprender Streamlit

https://demost-1.streamlit.app/
---

## 💻 Cómo correr local

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 🐳 Con Docker (producción)

```bash
docker compose up --build -d
```

### 🧹 Detener

```bash
docker compose down
```

---

## 🧑‍💻 Con Docker (desarrollo / hot-reload)

```bash
docker compose -f docker-compose.dev.yml up --build
```

---

## 🧰 Entrar al contenedor (bash)

```bash
docker ps    # para ver el <container_id>
docker exec -it <container_id> bash
```

---

## 🧼 Limpiar imágenes y contenedores antiguos

```bash
docker system prune -f
```
