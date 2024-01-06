// app.js
document.getElementById('upload-form').addEventListener('submit', function(event) {
    event.preventDefault();

    // Clear the previous images and display the loading text
    var resultDiv = document.getElementById('result');
    resultDiv.innerHTML = '正在生成图片...';

    // Create a new FormData object
    var formData = new FormData();

    // Get the current file from the file input field
    var file = document.getElementById('image-upload').files[0];

    // If there's no file, return early
    if (!file) {
        return;
    }

    // Append the file to the FormData object
    formData.append('image', file);

    // Append the width and height values to the FormData object
    formData.append('width', document.getElementById('width').value);
    formData.append('height', document.getElementById('height').value);
    formData.append('steps', document.getElementById('steps').value);

    // Send the form data to the server
    fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(images => {
        // Clear the loading text
        resultDiv.innerHTML = '';

        // Display the new images
        images.forEach((image, index) => {
            var imgElement = document.createElement('img');
            imgElement.src = 'data:image/jpeg;base64,' + image;
            resultDiv.appendChild(imgElement);
        });
    })
    .catch(error => {
        console.error('There has been a problem with your fetch operation:', error);
    });
});

document.getElementById('image-upload').addEventListener('change', function(event) {
    var file = event.target.files[0];
    var reader = new FileReader();
    reader.onload = function(e) {
        var imgElement = document.createElement('img');
        imgElement.src = e.target.result;

        var originalDiv = document.getElementById('original');
        originalDiv.innerHTML = '';
        originalDiv.appendChild(imgElement);
    };
    reader.readAsDataURL(file);
});