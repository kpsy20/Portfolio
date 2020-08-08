using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
public class InformationScript : MonoBehaviour {
    public Text yout, back, explain;
	// Use this for initialization
	// Update is called once per frame
	void Update () {
        if (Application.platform == RuntimePlatform.Android)
        {
            if (Input.GetKey(KeyCode.Escape))
            {
                SceneManager.LoadScene("Menu");
            }
        }
        if(ButtonController.lang == true)
        {
            yout.text = "유튜브 링크";
            back.text = "뒤로 가기";
            explain.text = "개발자: 햄붐" + "\n\n" + "Youtube채널"+"\n"+ "햄붐의 꾸물꾸물" + "\n\n" + "위 유튜브에서 개발 후기 +" + "\n" + "홍보영상등을 보실 수" + "\n" + "있습니다!"
                + "\n" + "\n" + "아래 버튼을 누르시면" + "\n" + "유튜브로 이동합니다.";
        }
        if(ButtonController.lang == false)
        {
            yout.text = "YouTube Link";
            back.text = "Go to Menu";
            explain.text = "Developer: HamBum" + "\n\n" + "Youtube channel"+"\n"+ "햄붐의 꾸물꾸물" + "\n\n" + "You can see the video of the development +" + "\n" + "the promotion in the above YouTube link.";
        }
    }
}
