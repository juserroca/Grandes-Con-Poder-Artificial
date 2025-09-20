from django import forms

class UploadCSVForm(forms.Form):
    csv_file = forms.FileField(
        label="📂 Archivo CSV"
    )
    action = forms.ChoiceField(
        label="⚙️ ¿Qué deseas hacer?",
        choices=[('train', 'Reentrenar Modelo'), ('predict', 'Predecir')],
        widget=forms.RadioSelect
    )
