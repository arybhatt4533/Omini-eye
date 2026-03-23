// Page load hote hi chalega
window.onload = function () {

    const data = JSON.parse(localStorage.getItem("result"));
    const resultBox = document.getElementById("resultBox");

    if (!data) {
        resultBox.innerHTML = "<p>No data found</p>";
        return;
    }

    // Agar match mila
    if (data.match === true) {
        resultBox.innerHTML = `
            <h3 style="color: green;">✅ Match Found</h3>
            <p><b>Name:</b> ${data.name || "Unknown"}</p>
            <p><b>Location:</b> ${data.location || "Unknown"}</p>
        `;
    } else {
        resultBox.innerHTML = `
            <h3 style="color: red;">❌ No Match Found</h3>
        `;
    }
};

// Back button function
function goBack() {
    window.location.href = "upload.html";
}