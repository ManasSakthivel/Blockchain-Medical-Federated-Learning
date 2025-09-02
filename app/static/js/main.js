// Main JavaScript for EHR Blockchain System

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Form validation
    setupFormValidation();

    // Blockchain status indicator
    setupBlockchainStatus();

    // Auto-hide alerts
    setupAutoHideAlerts();

    // Smooth scrolling for anchor links
    setupSmoothScrolling();

    // Table row highlighting
    setupTableRowHighlighting();
});

// Form validation setup
function setupFormValidation() {
    const forms = document.querySelectorAll('.needs-validation');
    
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });
}

// Blockchain status indicator
function setupBlockchainStatus() {
    // This would be connected to your blockchain service
    // For now, we'll simulate the status
    const statusIndicator = document.getElementById('blockchain-status');
    
    if (statusIndicator) {
        // Simulate blockchain connection check
        setTimeout(() => {
            const isConnected = Math.random() > 0.3; // 70% chance of being connected
            updateBlockchainStatus(isConnected);
        }, 1000);
    }
}

function updateBlockchainStatus(isConnected) {
    const statusIndicator = document.getElementById('blockchain-status');
    
    if (statusIndicator) {
        if (isConnected) {
            statusIndicator.className = 'blockchain-status connected';
            statusIndicator.innerHTML = '<i class="fas fa-link me-1"></i>Blockchain Connected';
        } else {
            statusIndicator.className = 'blockchain-status disconnected';
            statusIndicator.innerHTML = '<i class="fas fa-unlink me-1"></i>Blockchain Disconnected';
        }
    }
}

// Auto-hide alerts
function setupAutoHideAlerts() {
    const alerts = document.querySelectorAll('.alert');
    
    alerts.forEach(alert => {
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000); // Auto-hide after 5 seconds
    });
}

// Smooth scrolling
function setupSmoothScrolling() {
    const links = document.querySelectorAll('a[href^="#"]');
    
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Table row highlighting
function setupTableRowHighlighting() {
    const tableRows = document.querySelectorAll('tbody tr');
    
    tableRows.forEach(row => {
        row.addEventListener('mouseenter', function() {
            this.style.backgroundColor = '#f8f9fa';
        });
        
        row.addEventListener('mouseleave', function() {
            this.style.backgroundColor = '';
        });
    });
}

// Medical record functions
function viewMedicalRecord(recordId) {
    // This would fetch and display medical record details
    console.log('Viewing medical record:', recordId);
    
    // You could implement a modal or redirect to a detailed view
    window.location.href = `/patient/record/${recordId}`;
}

// Consultation functions
function bookConsultation() {
    // This would open the consultation booking form
    console.log('Opening consultation booking form');
    
    // You could implement a modal or redirect to booking page
    window.location.href = '/patient/book-consultation';
}

// Doctor functions
function addMedicalRecord(patientId) {
    // This would open the medical record form
    console.log('Adding medical record for patient:', patientId);
    
    // You could implement a modal or redirect to record form
    window.location.href = `/doctor/add-record/${patientId}`;
}

// Admin functions
function addDoctor() {
    // This would open the doctor registration form
    console.log('Opening doctor registration form');
    
    // You could implement a modal or redirect to registration page
    window.location.href = '/admin/add-doctor';
}

// Utility functions
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        showToast('Copied to clipboard!', 'success');
    }, function(err) {
        console.error('Could not copy text: ', err);
        showToast('Failed to copy to clipboard', 'error');
    });
}

function showToast(message, type = 'info') {
    // Create toast element
    const toastContainer = document.getElementById('toast-container') || createToastContainer();
    
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');
    
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
    `;
    
    toastContainer.appendChild(toast);
    
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
    
    // Remove toast element after it's hidden
    toast.addEventListener('hidden.bs.toast', function() {
        toast.remove();
    });
}

function createToastContainer() {
    const container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'toast-container position-fixed top-0 end-0 p-3';
    container.style.zIndex = '1055';
    document.body.appendChild(container);
    return container;
}

// Blockchain integration functions (placeholder)
function connectToBlockchain() {
    console.log('Connecting to blockchain...');
    // This would integrate with your blockchain service
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            const success = Math.random() > 0.3;
            if (success) {
                updateBlockchainStatus(true);
                resolve(true);
            } else {
                updateBlockchainStatus(false);
                reject(new Error('Failed to connect to blockchain'));
            }
        }, 2000);
    });
}

// IPFS integration functions (placeholder)
function uploadToIPFS(file) {
    console.log('Uploading to IPFS:', file.name);
    // This would integrate with your IPFS service
    return new Promise((resolve) => {
        setTimeout(() => {
            const hash = 'Qm' + Math.random().toString(36).substring(2, 15);
            resolve(hash);
        }, 1000);
    });
}

// Export functions for use in other scripts
window.EHRApp = {
    viewMedicalRecord,
    bookConsultation,
    addMedicalRecord,
    addDoctor,
    copyToClipboard,
    showToast,
    connectToBlockchain,
    uploadToIPFS
}; 