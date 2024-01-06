document.getElementById('upload-form').addEventListener('submit', function(event) {
    event.preventDefault();
    var formData = new FormData();
    formData.append('image', document.getElementById('image-upload').files[0]);
    formData.append('width', document.getElementById('width').value);
    formData.append('height', document.getElementById('height').value);
    fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(images => {
        var resultDiv = document.getElementById('result');
        resultDiv.innerHTML = ''; // Clear previous images
        images.forEach((image, index) => {
            var imgElement = document.createElement('img');
            imgElement.src = 'data:image/jpeg;base64,' + image;
            imgElement.addEventListener('click', function() {
                var confirmDownload = confirm('Do you want to download this image?');
                if (confirmDownload) {
                    var link = document.createElement('a');
                    link.href = imgElement.src;
                    link.download = `resized_image_${index}.jpeg`;
                    link.click();
                }
            });
            resultDiv.appendChild(imgElement);
        });
    })
    .catch(error => console.error('Error:', error));
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

        // Call the generateImages function when an image is uploaded
        generateImages();
    };
    reader.readAsDataURL(file);
});

function generateImages() {
    // Start the loading bar at 0%
    document.getElementById('loading-bar').style.width = '0%';

    fetch('/generate-images', {
        method: 'POST',
        // ...other options...
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        // Update the loading bar to 100% when the images have been generated
        document.getElementById('loading-bar').style.width = '100%';
    })
    .catch(error => {
        console.error('There has been a problem with your fetch operation:', error);
    });
}