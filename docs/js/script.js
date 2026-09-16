
document.addEventListener('DOMContentLoaded', () => {
    // Animate bars
    const bars = document.querySelectorAll('.bar-fill');
    bars.forEach(bar => {
        const target = bar.getAttribute('data-width');
        setTimeout(() => {
            bar.style.width = target;
        }, 300);
    });
});
