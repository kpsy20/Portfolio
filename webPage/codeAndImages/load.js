var seconds=0;
function startTimer()
{
  window.setInterval("updateTime()", 1000)
}
function updateTime()
{
  ++seconds;
  document.getElementById("soFar").innerHTML = seconds;
}
window.addEventListener("load", startTimer, false);
