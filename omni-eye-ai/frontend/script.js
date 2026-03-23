async function uploadImage() {
    const fileInput = document.getElementById("imageInput");
    const file = fileInput.files[0];

    if (!file) {
        alert("Select file first");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("http://127.0.0.1:8000/upload", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        // result page pe bhejo
        localStorage.setItem("result", JSON.stringify(data));
        window.location.href = "result.html";

    } catch (error) {
        alert("Error connecting backend");
    }
}