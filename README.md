# 🌌 Grandes-Con-Poder-Artificial 🚀

Bienvenido a **MLApp**, una aplicación en **Django + Machine Learning** que te permite cargar datasets en CSV y realizar dos acciones principales:

- 🔄 **Reentrenar un modelo**
- 🔮 **Predecir con datos nuevos**

---

## ✨ Características

- 📂 Carga de archivos CSV mediante un formulario amigable
- 🧠 Entrenamiento y reentrenamiento de modelos de ML
- 🔎 Predicción sobre nuevos datos
- 🎨 Interfaz web sencilla con Django Templates

---

## ⚙️ Instalación

1. Clona este repositorio:

```bash
git clone https://github.com/juserroca/Grandes-Con-Poder-Artificial.git
cd mlapp
```

2. Crea y activa un entorno virtual:

```bash
conda create -n astro-ml python=3.13 -y
conda activate astro-ml
```

3. Instala las dependencias:

```bash
pip install -r requirements.txt
```

4. Realiza las migraciones de Django:

```bash
python manage.py migrate
```

5. Inicia el servidor:

```bash
python manage.py runserver
```

---

## 📊 Uso

1. Abre en tu navegador [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
2. Carga un archivo CSV con tus datos
3. Selecciona una acción:
   - 🔄 **Reentrenar modelo**
   - 🔮 **Predecir**
4. Obtén los resultados directamente en la web 🎉

---

## 📂 Estructura del proyecto

```
mlapp/
├── mlapp/                # Configuración principal de Django
├── templates/            # Plantillas HTML
├── static/               # Archivos estáticos (CSS, JS, imágenes)
├── views.py              # Lógica de las vistas
├── forms.py              # Formularios (UploadCSVForm)
├── models.py             # Modelos de Django (si aplica)
└── ...
```

---

## 🛠 Tecnologías usadas

- 🐍 Python 3.13
- 🌐 Django 5.2.6
- 📦 Scikit-learn (u otra librería de ML)
- 📊 Pandas / Numpy

---

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! 🎉

1. Haz un fork del repositorio
2. Crea una rama con tu feature/fix
3. Haz un Pull Request

---

## 📜 Licencia

Este proyecto está bajo la licencia **MIT**.

---

💡 Hecho con pasión por los datos y el universo ✨🪐
