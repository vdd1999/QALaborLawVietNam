function formatResMess(str) {
  // Handle cases where the input is empty or null
  if (!str) {
    return "";
  }

  // Convert the first character to uppercase and the rest to lowercase
  return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
}

function createChatUser(message) {
  const liEle = $("<li>").addClass("d-flex justify-content-between px-3 mb-4");
  const cardEle = $("<div>").addClass("card w-100");
  const cardBodyEle = $("<div>").addClass("card-body");
  const pEle = $("<p>").addClass("mb-0").text(message);
  cardBodyEle.append(pEle);
  cardEle.append(cardBodyEle);
  liEle.append(cardEle);
  const imgEle = $("<img>")
    .attr("src", "/static/images/user.png")
    .attr("alt", "avatar")
    .addClass(
      "rounded-circle bg-light d-flex align-self-start ms-3 shadow-1-strong"
    )
    .attr("width", "60");
  liEle.append(imgEle);
  return liEle;
}

function createChatBot({
  groupId = "",
  id = "",
  message = "Vui lòng chờ đợi",
  isAnswer = false,
}) {
  const liEle = $("<li>").addClass("d-flex justify-content-between px-3 mb-4");
  liEle.attr("data-group-id", groupId);
  liEle.attr("data-id", id);
  const cardEle = $("<div>").addClass("card w-100");
  const cardBodyEle = $("<div>").addClass("card-body d-flex");
  const pEle = $("<p>")
    .addClass("mb-0 flex-grow-1")
    .text(formatResMess(message));
  const imgEle = $("<img>")
    .attr("src", "/static/images/bot.webp")
    .attr("alt", "avatar")
    .addClass(
      "rounded-circle bg-light d-flex align-self-start me-3 shadow-1-strong"
    )
    .attr("width", "60");
  liEle.append(imgEle);
  cardBodyEle.append(pEle);
  if (isAnswer) {
    const iEle = $("<i>")
      .addClass("fa fa-commenting text-success")
      .attr("title", "Show answer")
      .attr("data-id", id)
      .attr("data-group-id", groupId)
      .attr("onclick", "showAnswer(this);");
    cardBodyEle.append(iEle);
  }
  cardEle.append(cardBodyEle);
  liEle.append(cardEle);
  return liEle;
}

function createLoadingEle() {
  const liEle = $("<li>")
    .addClass("d-flex justify-content-between px-3 mb-4")
    .attr("id", "loading");
  const imgEle = $("<img>")
    .attr("src", "/static/images/bot.webp")
    .attr("alt", "avatar")
    .addClass(
      "rounded-circle bg-light d-flex align-self-start me-3 shadow-1-strong"
    )
    .attr("width", "60");
  liEle.append(imgEle);
  const cardEle = $("<div>").addClass("card w-100");
  const cardBodyEle = $("<div>").addClass(
    "card-body d-flex align-items-center"
  );
  const spinnerEle = $("<div>")
    .addClass("spinner-grow text-secondary me-1")
    .css("width", "1rem")
    .css("height", "1rem")
    .attr("role", "status");
  const spanEle = $("<span>").addClass("visually-hidden").text("Loading...");
  spinnerEle.append(spanEle);
  const lstSpinner = [
    spinnerEle.clone(),
    spinnerEle.clone(),
    spinnerEle.clone(),
  ];
  lstSpinner.map((item) => {
    cardBodyEle.append(item);
  });
  cardEle.append(cardBodyEle);
  liEle.append(cardEle);
  return liEle;
}

function showAnswer(ele) {
  const id = $(ele).attr("data-id");
  const groupId = $(ele).attr("data-group-id");
  if (!groupId) {
    const dataStorage = JSON.parse(sessionStorage.getItem("messData"));
    const loadingEle = createLoadingEle();
    const chatContainer = $("#chatContainer");
    chatContainer.append(loadingEle);
    const chatBotReply = createChatBot({
      message: dataStorage?.answer?.text,
      isAnswer: false,
    });
    $("#loading").remove();
    chatContainer.append(chatBotReply);

    console.log(dataStorage);
  } else {
    const dataStorage = JSON.parse(sessionStorage.getItem("groupData"));
    console.log(dataStorage);
  }
}
