document.addEventListener("click", (event) => {
  const addButton = event.target.closest("[data-add-record]");
  if (addButton) {
    const kind = addButton.dataset.addRecord;
    const container = document.getElementById(`${kind}-records`);
    const template = document.getElementById(`${kind}-record-template`);
    const card = template.content.firstElementChild.cloneNode(true);
    const numberInput = card.querySelector("[data-number-input]");
    numberInput.value = String(container.querySelectorAll("[data-record-card]").length + 1);
    container.append(card);
    card.querySelector("textarea")?.focus();
    renumberCards(container);
  }

  const deleteButton = event.target.closest(".delete-record");
  if (deleteButton) {
    const card = deleteButton.closest("[data-record-card]");
    const container = card.parentElement;
    card.remove();
    renumberCards(container);
  }
});

document.addEventListener("change", (event) => {
  if (event.target.matches("input[type=file]")) {
    const file = event.target.files?.[0];
    if (file) {
      const label = event.target.closest(".drop-zone");
      label?.classList.add("has-file");
      const prompt = label?.querySelector("strong");
      if (prompt) prompt.textContent = file.name;
    }
  }
});

function renumberCards(container) {
  container.querySelectorAll("[data-record-card]").forEach((card, index) => {
    const visible = card.querySelector("[data-visible-number]");
    if (visible) visible.textContent = String(index + 1);
    const input = card.querySelector("[data-number-input]");
    if (input && input.type === "hidden") input.value = String(index + 1);
  });
}