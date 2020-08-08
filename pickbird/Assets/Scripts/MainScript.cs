using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
public class MainScript : MonoBehaviour {
    public Text explain, totaltext, selecttext;

    // Update is called once per frame
    void Update() {
        if (Application.platform == RuntimePlatform.Android)
        {
            if (Input.GetKey(KeyCode.Escape))
            {
                SceneManager.LoadScene("Menu");
            }
        }
        if(ButtonController.lang == true)
        {
            explain.text = "숫자만 입력해 주세요!";
            totaltext.text = "총 인원 (9999이하)";
            selecttext.text = "당첨 인원 (999이하)";
            totaltext.fontSize = 80;
            selecttext.fontSize = 80;
        }
        if(ButtonController.lang == false)
        {
            explain.text = "Only number please!";
            totaltext.text = "Total number of people."+"\n"+"(Under 9999)";
            totaltext.fontSize = 50;
            selecttext.text = "Number to be selected."+"\n"+"(Under 999)";
            selecttext.fontSize = 50;
        }
    }
}
		
