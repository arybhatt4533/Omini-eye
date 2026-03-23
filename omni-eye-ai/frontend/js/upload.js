async function uploadImage() {
    const fileInput = document.getElementById("imageInput");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select an image");
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

        // Save result
        localStorage.setItem("result", JSON.stringify(data));

        // Go to dashboard
        window.location.href = "dashboard.html";

    } catch (error) {
        console.error(error);
        alert("Backend error");
    }
}