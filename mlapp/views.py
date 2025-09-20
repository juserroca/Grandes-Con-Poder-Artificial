from django.shortcuts import render
from .forms import UploadCSVForm
from . import utils

def upload_csv(request):
    result = None
    if request.method == "POST":
        form = UploadCSVForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES["csv_file"]
            action = form.cleaned_data["action"]

            if action == "train":
                result = utils.train_model(csv_file)
            elif action == "predict":
                result = utils.predict_with_model(csv_file).head(20).to_html(classes="table table-dark table-hover")
    else:
        form = UploadCSVForm()

    return render(request, "mlapp/upload.html", {"form": form, "result": result})
